# Paraphina Roadmap

This roadmap is the canonical plan to evolve Paraphina from the current
research/simulation backbone into the *canonical* “venue-agnostic perp MM + hedge”
engine described in:

- `docs/WHITEPAPER.md`
  - Part I: **Implementation truth / what exists today**
  - Part II: **Appendix: Canonical target spec (verbatim)**

The guiding rule is: **code and telemetry contracts come first**. Any spec claim
is “planned” until it is implemented, tested, and measured via the batch harness.

---

## 0. Non-negotiables

### 0.1 Determinism and replayability
- All strategy decisions must be deterministic given:
  - initial state,
  - a time-ordered input event stream,
  - the config/environment,
  - an explicit RNG seed if any randomness exists.

### 0.2 Research contracts are production contracts
The batch backbone depends on stable, machine-readable outputs:
- `batch_runs/orchestrator.py` runs many configs and returns a tidy DataFrame
  via `results_to_dataframe(...)`.
- `batch_runs/metrics.py` parses end-of-run stdout summaries using
  `parse_daily_summary(stdout)` and aggregates runs via `aggregate_basic(...)`.

Do not break these contracts without versioning.

### 0.3 Safety defaults win
If there is ambiguity between “trade more” and “reduce risk”, defaults must pick
“reduce risk” (especially around liquidation distance, hard limits, and kill).

---

## 1. Current baseline (observed in repo)

### 1.1 Generic run orchestrator exists
`batch_runs/orchestrator.py` provides:
- `EngineRunConfig(cmd, env, label, run_id, workdir)`
- `run_engine(...)` running subprocesses with env overlays
- `run_many(...)` sequential runner with Timeout handling
- `results_to_dataframe(...)` producing tidy per-run rows of `label + metrics`

This is the primary research execution interface.

### 1.2 Shared stdout metrics parser exists
`batch_runs/metrics.py` provides:
- `parse_daily_summary(stdout)` extracting:
  - `pnl_realised`, `pnl_unrealised`, `pnl_total`
  - `kill_switch` boolean
- `aggregate_basic(df_runs, group_keys)` producing grouped summaries
  similar to exp02/exp03.

### 1.3 Exp03 is the canonical “experiment script” pattern
`batch_runs/exp03_stress_search.py` demonstrates the pattern:
- load a prior summary CSV (`runs/exp02_profile_grid/exp02_profile_grid_summary.csv`)
- derive “profile centres”
- build a grid of `EngineRunConfig` with env overrides (profile, q0, etc.)
- run via `run_many(...)`
- write:
  - per-run CSV
  - grouped summary CSV

---

## 2. Milestone plan (spec-alignment driven)

### Milestone A — Documentation + “contracts first”
**Goal:** make it impossible to confuse “planned spec” with “implemented behavior”.

Deliverables:
- `docs/WHITEPAPER.md` is the single entry point:
  - “Implementation truth” section stays conservative and code-backed
  - “Canonical spec” is preserved verbatim
- Add/maintain a short “Drift Register” section listing:
  - spec features not implemented yet,
  - known mismatches in naming/semantics (e.g., Warning/Critical vs Warning/HardLimit).

Acceptance criteria:
- A new contributor can run exp03 and understand:
  - what the binary expects (env knobs),
  - what stdout must contain for parsing,
  - what CSVs are produced.

---

### Milestone B — Standardise knobs and experimentability
**Goal:** every strategy parameter that matters is:
- configurable (env/config),
- logged/observable,
- sweepable via orchestrator labels.

Deliverables:
- A canonical env var list in docs (and used consistently):
  - examples already referenced in `batch_runs/orchestrator.py` docstring:
    - `PARAPHINA_RISK_PROFILE`
    - `PARAPHINA_INIT_Q_TAO`
    - `PARAPHINA_HEDGE_BAND_BASE`
    - `PARAPHINA_HEDGE_MAX_STEP`
    - `PARAPHINA_MM_SIZE_ETA`
    - `PARAPHINA_VOL_REF`
    - `PARAPHINA_DAILY_LOSS_LIMIT`
- A single authoritative end-of-run stdout summary format that
  `parse_daily_summary` parses (and unit tests for the regex).

Acceptance criteria:
- `batch_runs/metrics.py` has tests with representative stdout blocks.
- exp scripts can be added without inventing new parsing logic.

---

### Milestone C — Risk regime correctness + kill switch semantics
**Goal:** risk states match the canonical model: Normal / Warning / Critical
with a kill switch that is explicit and unambiguous.

Deliverables:
- Define regime semantics and thresholds in one place.
- Ensure **Critical implies kill_switch** behaviorally:
  - cancel/stop new orders,
  - allow only bounded “best-effort” risk reduction if implemented.

Acceptance criteria:
- A “Critical” condition in simulation produces:
  - `kill_switch=true` in stdout summary
  - (optionally) an explicit reason code
- A regression test covers:
  - PnL hard breach,
  - delta/basis hard breach,
  - liquidation-distance hard breach (if the engine models it).

---

### Milestone D — Fair value + volatility gating (robustness)
**Goal:** fair value and volatility estimates are stable, and strategy behavior
degrades safely when data quality drops.

Deliverables:
- FV estimator gating policy (stale books / outliers / min healthy venues).
- Vol floor behavior documented and tested.
- Telemetry includes:
  - fair value (or a “fv_available” flag),
  - effective volatility,
  - list/count of healthy venues used.

Acceptance criteria:
- With intentionally stale/missing venue data in sim:
  - FV update degrades gracefully
  - quoting shrinks or pauses depending on config.

---

### Milestone E — Cross-venue exits (canonical Section 12)
**Goal:** implement cross-perp exit optimization that:
- uses net edge after fees/slippage/vol buffers,
- incorporates basis/funding differentials,
- penalises basis risk increase and fragmentation.

Deliverables:
- A discrete “exit allocator” module callable after fill batches.
- A consistent definition of:
  - fragmentation score,
  - basis exposure change approximation,
  - effective edge threshold(s).

Acceptance criteria:
- A synthetic multi-venue scenario demonstrates:
  - exit chooses best levels first,
  - exits reduce basis/fragmentation when edges are similar,
  - exit allocator respects lot sizes and min notional.

---

### Milestone F — Global hedge allocator (canonical Section 13)
**Goal:** hedging is a *global* optimization over allowed venues:
- LQ controller decides global step size with deadband,
- allocation chooses cheapest/least-risk venues first,
- constraints include funding/basis/margin/liquidation/fragmentation.

Deliverables:
- Hedge allocator as a first-class component:
  - input: desired global ΔH, venue snapshots
  - output: per-venue IOC intents with guard prices
- Costs modeled as additive components:
  - immediate execution + fee + slippage buffer
  - funding carry preference
  - basis exposure effect
  - liquidation distance penalty
  - fragmentation penalty

Acceptance criteria:
- In an adversarial test:
  - allocator avoids venues near liquidation
  - allocator prefers better funding/basis when execution is comparable
  - allocator respects global max-step and per-venue max hedgeable size

---

### Milestone G — Multi-venue quoting model (canonical Section 9–11)
**Goal:** quoting per venue reflects:
- Avellaneda–Stoikov baseline,
- basis/funding shifts,
- per-venue inventory targets,
- toxicity and liquidation-distance-aware size constraints,
- stable order management (min quote lifetime, tolerance thresholds).

Deliverables:
- Explicit functions with unit tests for:
  - reservation price components
  - half-spread computation
  - edge filters
  - size constraints and shrink logic
  - cancel/replace logic thresholds

Acceptance criteria:
- Under the same sim tape:
  - behavior is stable across refactors
  - “Warning” widens and caps
  - “Critical” halts new quoting

---

### Milestone H — Production-ready separation (I/O vs strategy)
**Goal:** preserve a clean boundary:
- strategy core is pure/deterministic
- I/O layer (WS/REST) is async and replaceable
- execution gateway handles venue semantics (TIF/post-only/IOC guard).

Deliverables:
- Action interface: `PlaceOrder`, `CancelOrder`, `CancelAll`, `SetKillSwitch`, …
- Venue adapter trait per exchange
- Rate limiting and retry policies (configurable)

Acceptance criteria:
- Full replay test:
  - feed recorded events,
  - reproduce identical action stream.

---

## 3. Experiment roadmap (batch_runs)
This repo already has a strong experiment pattern. Extend it deliberately.

### Recommended experiment ladder
- exp02: profile grid sweeps → “centres”
- exp03: stress vs starting inventory (`INIT_Q_TAO_GRID`) per profile centre
- exp04+: add funding/basis/liquidation perturbations once modeled in sim
- exp05+: allocator A/Bs (exit allocator variants; hedge allocator variants)

### Standard outputs
Every experiment should write under `runs/<exp_name>/`:
- `*_runs.csv` — per-run rows (labels + parsed metrics)
- `*_summary.csv` — grouped summaries

---

## 4. Definition of Done for any feature
A feature is “done” only when:
1) It is implemented behind config if risky,
2) It has unit tests for core math,
3) It is observable (telemetry + stdout summary if needed),
4) It has an experiment that demonstrates improvement or safety,
5) It is documented in `docs/WHITEPAPER.md` drift register (until drift is removed).

---

## RL and GPU Training Track

This track evolves Paraphina from a deterministic strategy into a GPU-trained
reinforcement learning policy, while keeping hard safety controls in Rust.

### RL-0: Foundations (interfaces + instrumentation)
**Goal:** make the current engine “RL-ready” without changing behaviour.

- [ ] Add a versioned `Observation` schema derived from `GlobalState` + venues
- [ ] Add a `Policy` interface with a default `HeuristicPolicy` (current logic)
- [ ] Log policy inputs/outputs in telemetry (obs_version, policy_version)
- [ ] Add deterministic seeding + episode reset mechanics in simulation mode
- [ ] Add a “shadow mode” runner (policy proposes, baseline executes)

**Exit criteria**
- Replays are deterministic and byte-for-byte reproducible
- Telemetry is sufficient to reconstruct rewards and constraints

### RL-1: Gym-style environment and vectorised simulation
**Goal:** turn the simulator into a high-throughput training environment.

- [ ] Create `paraphina_env` wrapper (Python bindings) around Rust sim
- [ ] Support vectorised rollouts (N environments in parallel)
- [ ] Add domain randomisation knobs:
  - fees, spreads, slippage model
  - funding regimes
  - volatility regimes
  - venue staleness / disable events

**Exit criteria**
- Sustained high-step throughput (enough to saturate GPU training)
- Training/eval parity: same reward + constraints computed everywhere

### RL-2: Imitation learning baseline (behaviour cloning)
**Goal:** train a policy to imitate the heuristic strategy.

- [x] Generate large trajectory datasets from heuristic policy
- [x] Train BC policy on bounded "control surface" actions (spread/size/offset)
- [x] Evaluate: action error, PnL parity, risk parity under randomisation

**Exit criteria**
- BC policy matches baseline within tolerance and does not increase kill rate

**Implementation (Completed):**
- `src/rl/action_encoding.rs`: Versioned action encoding (ACTION_VERSION=1) with bounded ranges
- `src/rl/trajectory.rs`: TrajectoryCollector for deterministic dataset generation
- `paraphina_env`: Python bindings for TrajectoryCollector class
- `python/rl2_bc/`: Dataset generation, BC training (PyTorch MLP), and evaluation scripts
- Full test coverage for encoding determinism and trajectory reproducibility

### RL-3: GPU RL baseline (robust)
**Goal:** safely improve on BC using online RL in simulation.

- [ ] PPO baseline on control-surface actions
- [ ] Add constrained RL (Lagrangian penalties) to keep budgets:
  kill_prob, drawdown, basis/delta usage
- [ ] Continuous evaluation suite + regression gates

**Exit criteria**
- Measurable improvement in risk-adjusted score with no budget regressions

### RL-4: Model-based RL (advanced)
**Goal:** improve sample efficiency and robustness.

- [ ] Train a learned world model from trajectories
- [ ] Dreamer-style training / imagination rollouts
- [ ] Always validate final candidates in “true” simulator (anti-exploitation)

**Exit criteria**
- Consistent improvements across random seeds and stress scenarios

### RL-5: Shadow deployment and safety validation
**Goal:** production-grade validation without trading risk.

- [ ] Shadow inference in prod (policy proposes, baseline executes)
- [ ] Counterfactual evaluation via replay
- [ ] Latency + failure-mode testing (timeouts, invalid outputs)
- [ ] Model card + deployment checklist

**Exit criteria**
- Stable inference, stable decisions, no safety constraint violations in shadow

### RL-6: Limited live execution
**Goal:** controlled live rollout.

- [ ] Start with tiny caps + strict kill thresholds
- [ ] Gradually increase caps only if budgets remain satisfied
- [ ] Continuous monitoring and automatic rollback to baseline

**Exit criteria**
- Sustained alignment with risk budgets in live conditions

## Long-term: “Fully Optimised Strategy” Track (Quant Optimisation → GPU RL)

This repo evolves in three layers:
1) deterministic baseline strategy (production-safe),
2) automated quant optimisation (search + Monte Carlo + adversarial stress),
3) GPU-trained RL policies behind a hard deterministic safety shield.

### Phase A — Quant Optimisation Foundation (pre-RL, highest ROI)
**Goal:** improve robustness and performance without introducing ML risk.

- [ ] Scenario library (seeded, reproducible)
  - volatility regimes, spread/depth shocks, venue outages, funding inversions, basis spikes
  - latency / partial fill / cancel storm modelling
- [ ] Monte Carlo runner at scale
  - generate thousands–millions of scenarios
  - report tail risk metrics: VaR/CVaR, worst-quantile drawdown, kill probability confidence intervals
- [ ] Adversarial / worst-case search
  - CEM / evolutionary search over scenario parameters to find failures quickly
  - promote “failure seeds” to a permanent regression suite
- [ ] Multi-objective tuning of strategy knobs
  - Bayesian optimisation / CMA-ES / evolutionary search
  - constraints enforced by risk-tier budgets (kill_prob, drawdown, min pnl)
- [ ] Promotion pipeline
  - only promote presets that pass out-of-sample + adversarial regression suites

### Phase B — World Model (Learned Simulator) on GPUs
**Goal:** learn a high-fidelity dynamics model from telemetry so RL is sample-efficient.

- [ ] Telemetry schema stabilisation (what the world model needs)
  - observations: books, spreads/depth, funding, basis, fills, latency proxies
  - actions: quotes, cancels, exits, hedges
  - outcomes: fills, slippage, markouts, pnl, risk events
- [ ] World model training pipeline (offline)
  - ensemble models + uncertainty estimation
  - domain randomisation hooks
- [ ] Evaluation
  - compare world-model rollouts vs true simulator / historical telemetry
  - reject if model error increases tail risk estimates

### Phase C — Safe RL (GPU) behind deterministic risk shield
**Goal:** RL improves execution/hedging/quoting while never violating invariants.

- [ ] Define “policy surfaces” (bounded control only)
  - quote spread/size multipliers, small skew shifts, hedge allocator weights, exit prioritisation
- [ ] Implement safety shield
  - delta/basis/liquidation/daily-loss constraints enforced deterministically
  - kill-switch remains deterministic
- [ ] RL training
  - model-based RL (Dreamer/MuZero-style) + constrained RL
  - offline RL + conservative objectives
- [ ] Shadow mode in live gateway
  - policy proposes actions, baseline executes
  - log counterfactuals and divergence metrics
- [ ] Gated execution rollout
  - tight caps, gradual expansion only after passing safety scorecards

### Definition of Done (for “Fully Optimised”)
A strategy revision is “fully optimised” only if it:
- improves out-of-sample tail metrics (CVaR / worst-quantile drawdown),
- reduces kill probability with statistical confidence,
- passes adversarial regression suite,
- preserves determinism and replayability,
- keeps safety invariants as the final authority (even with RL enabled).