# AI Playbook for Paraphina

This playbook is for using an AI assistant (LLM) to make safe, testable,
reviewable changes in Paraphina.

It is intentionally *repo-specific* and focuses on the experiment backbone in
`batch_runs/` plus the spec-alignment process described in `docs/WHITEPAPER.md`.

---

## 1. First rules (don’t skip)

### 1.1 Never invent code
If you (or the AI) haven’t opened a file, do not claim what it contains.
When in doubt:
- ask the AI to locate the symbol with grep,
- or paste the relevant file snippet.

### 1.2 Preserve research contracts
The batch system expects:
- a stable stdout summary (parsed by `batch_runs/metrics.py`)
- stable run orchestration (driven by `batch_runs/orchestrator.py`)
- CSV outputs with consistent column naming.

Breaking these without versioning will silently invalidate experiments.

### 1.3 Determinism over cleverness
Strategy logic must remain deterministic with:
- fixed config,
- fixed input events,
- fixed seed (if any RNG exists).

### 1.4 Safety-first bias
In ambiguous situations, prefer:
- smaller sizes,
- wider spreads,
- faster risk reduction,
- earlier kill.

---

## 2. Repo primitives the AI should understand

### 2.1 Orchestrator: running the binary many times
`batch_runs/orchestrator.py` defines the run contract:

- `EngineRunConfig`
  - `cmd`: list of args (you pass the exact command)
  - `env`: env var overrides merged on top of current env
  - `label`: arbitrary JSON-serialisable metadata merged into output DF
  - `run_id`, `workdir`

- `run_many(configs, parse_metrics, timeout_sec, verbose)`
  - sequentially runs configs (good for reproducible research)

- `results_to_dataframe(results)`
  - flattens `label + metrics` into a tidy per-run DataFrame

**AI instruction:** when adding a new experiment, follow this pattern exactly.

### 2.2 Stdout parser: what the Rust binary must print
`batch_runs/metrics.py` parses a human-readable end-of-run summary.

It expects lines like:
- `Daily PnL (realised / unrealised / total): +123.45 / -67.89 / +55.56`
- `Kill switch: true` (or `Kill switch active: false`)

**AI instruction:** if you change the summary text, update the regex and
add tests. Otherwise experiments will silently lose metrics.

### 2.3 Experiment pattern (exp03 as template)
`batch_runs/exp03_stress_search.py` is the reference template:

1) load prior experiment output
2) derive a parameter “centre”
3) build a grid of `EngineRunConfig` env overlays
4) run via `run_many`
5) write:
   - `*_runs.csv`
   - `*_summary.csv`

**AI instruction:** copy this structure for new exp scripts.

---

## 3. Safe AI workflow for making changes

### Step 0 — State the goal precisely
Bad: “improve hedging”
Good: “implement per-venue hedge allocation cost terms for funding and
liquidation-distance penalty; validate via exp03-like sweep”

### Step 1 — Identify the source of truth
Ask the AI to locate:
- the strategy/engine code that decides quotes/hedges,
- the config/env parsing,
- the stdout summary printing,
- the telemetry schema.

### Step 2 — Identify the measurement hook
Before changing logic, confirm:
- which metric will improve (e.g., `pnl_total_mean`, `kill_switch_frac`)
- which experiment will detect it (existing exp03 or new exp)

### Step 3 — Propose a minimal patch
Ask the AI for:
- smallest possible diff,
- clear separation of pure logic vs I/O,
- explicit configuration knobs.

### Step 4 — Add tests for the math
For any numeric logic (skews, buffers, penalties, shrink factors):
- add unit tests for boundary cases and monotonicity:
  - “penalty increases as dist_liq_sigma decreases”
  - “hedge deadband shrinks as vol increases”
  - “Critical implies no new quotes”

### Step 5 — Run one small experiment
Use the orchestrator to validate:
- parsing works,
- CSV outputs are created,
- aggregated metrics make sense.

### Step 6 — Only then run full sweeps
Scale to bigger grids only after smoke tests.

---

## 4. Prompt templates that work well

### 4.1 “Locate the logic”
> Search the repo for where the engine prints the ‘Daily PnL’ summary and where
> kill switch is computed. Provide file paths and line ranges and explain the
> data flow from fills → pnl → stdout.

### 4.2 “Add a new knob”
> Add env var `PARAPHINA_EXIT_FRAGMENTATION_PENALTY` with default X.
> Ensure it is wired into config, logged in summary/telemetry, and included in
> experiment labels. Provide tests.

### 4.3 “Add an experiment”
> Create `batch_runs/expXX_name.py` that sweeps `PARAPHINA_HEDGE_MAX_STEP` across
> values [..] for profiles [..], uses `run_many`, writes per-run and summary
> CSVs, and prints top configurations by pnl_total_mean subject to
> kill_switch_frac <= Y.

### 4.4 “Don’t break the parser”
> Propose a change to stdout summary. Also update `batch_runs/metrics.py` regexes
> and add tests demonstrating both old and new formats are accepted.

---

## 5. Common pitfalls (and how to avoid them)

### 5.1 Sign conventions for loss limits
Make sure:
- a “daily loss limit” is treated consistently (usually a positive magnitude
  but compared against negative PnL).
- tests cover the boundary (exactly at limit).

### 5.2 Silent schema drift
If you rename:
- `pnl_total`, `kill_switch`, `risk_regime`, etc.
you must:
- update every consumer script,
- version the output.

### 5.3 Overfitting to one experiment
If a change improves exp03 but worsens robustness, require:
- at least one adversarial scenario (stale venues, high vol, funding shock).

### 5.4 Mixing strategy logic with venue semantics
Keep:
- strategy: compute desired actions in abstract terms
- gateway: enforce post-only/IOC/TIF/guard-bps semantics per venue

---

## 6. What “good” PRs look like
A PR is “good” when it includes:
- a clear problem statement and expected metric impact
- minimal diff
- tests for core math
- an experiment script or updated results showing improvement
- doc update:
  - update drift register in `docs/WHITEPAPER.md` if closing or introducing drift

---

## 7. Cross-links
- Canonical target algorithm + implementation-truth framing:
  - `docs/WHITEPAPER.md`
- Batch execution and parsing backbone:
  - `batch_runs/orchestrator.py`
  - `batch_runs/metrics.py`
  - `batch_runs/exp03_stress_search.py`

---

## Reinforcement Learning Playbook

This section describes the standard operating procedure for training,
validating, and deploying an RL policy for Paraphina.

### Non-negotiable invariants

The following MUST remain deterministic and enforced in Rust (not learned):

- Risk regime computation (Normal / Warning / HardLimit)
- Kill switch logic and kill behaviour
- Basis exposure computation and limits
- Liquidation-distance sizing guards
- Venue health gating (toxicity/staleness disables)
- Order validity (tick sizes, min sizes, post-only/tif constraints)

RL policies may only propose actions within a bounded “control surface”.
All outputs are clipped and validated before use.

### Policy contract

**Inputs:** a versioned `Observation` vector built from:
- global fair value, vol scalars, global inventory, basis exposures, PnL
- per-venue book features, funding, toxicity, position, liquidation distance
- risk regime flags

**Outputs (recommended):**
- per-venue `spread_scale_v`, `size_scale_v`, `rprice_offset_usd_v`
- global `hedge_scale`
- `hedge_venue_weights` over hedge-allowed venues

Outputs must be:
- bounded
- finite
- deterministic for the same observation (no RNG at inference)

### Reward standard

Reward must align with the world-model budget system:

- Primary: ΔPnL (mark-to-market) net of estimated costs
- Penalties: delta usage, basis usage, drawdown, toxicity, turnover
- Terminal: large penalty on kill-switch termination

All reward components should be reconstructable from telemetry.

### Training sequence

1) **Behaviour Cloning (BC)**
   - Train to imitate heuristic strategy actions on randomised sims
   - Produces a stable initial policy

2) **Offline RL (optional)**
   - Conservative improvements using logged trajectories

3) **Online RL (GPU)**
   - PPO (baseline) or SAC (continuous) on the control surface
   - Add constrained RL penalties for budget violations

4) **Model-based RL (advanced)**
   - Learned world model + imagination rollouts
   - Validate final policies in the true simulator

5) **Shadow deployment**
   - Policy runs in prod but does not trade
   - Compare decisions + replay counterfactuals

6) **Limited live rollout**
   - Tiny caps, strict budgets, auto rollback

### Evaluation gates

A policy is not eligible for deployment unless it passes:

- Budget alignment:
  - kill probability <= tier budget
  - mean drawdown magnitude <= tier budget
  - mean final pnl >= tier budget
- Stress tests:
  - high vol
  - venue disable / staleness cascades
  - funding regime flips
- Latency + reliability:
  - inference timeout behaviour verified
  - invalid output → safe fallback verified
- Reproducibility:
  - deterministic replay for a fixed seed and event stream

### Deployment format

Recommended:
- Export trained policy to ONNX
- Load and run in Rust with:
  - watchdog timeout
  - output validation + clipping
  - fallback to baseline heuristic policy
- Log `policy_version`, `obs_version`, and applied action after clamps

### Model card (required)

Every deployed model must include:
- training data summary (sims + date range)
- observation schema version
- action schema version
- reward definition
- budgets targeted (conservative/balanced/aggressive)
- evaluation results (tables + stress tests)
- known failure modes and mitigations

## RL Evolution Blueprint (GPU) — How We Upgrade Without Breaking Safety

This project upgrades the strategy in layers:
- Deterministic baseline (shipping engine)
- Automated quant optimisation (Monte Carlo + search)
- Safe RL (GPU) controlling bounded “policy surfaces” behind a deterministic shield

### 1) Non-negotiables (Safety and Determinism)
Even with RL enabled:
- Risk invariants are enforced deterministically (delta/basis/liquidation/daily-loss).
- Kill-switch remains deterministic.
- RL cannot directly place arbitrary orders; it can only:
  (a) modify bounded knobs, or
  (b) choose among already-safe candidate actions generated by deterministic logic.
- All runs remain replayable via (config_version_id, seed, event log).

### 2) Policy Surfaces (What RL Controls)
RL must not replace the whole strategy at once. It controls small interfaces:

**A) Quote policy (per venue)**
Outputs are bounded multipliers or small shifts:
- extra_spread_mult_v ∈ [0.5, 2.0]
- size_scale_v ∈ [0.0, 1.0]
- skew_shift_v (small TAO adjustment)
- soft-disable preference flags (hard constraints still apply)

**B) Hedge allocator**
Given a desired global hedge ΔH, RL outputs allocation weights across hedge-allowed venues:
- w_v ≥ 0, sum(w_v)=1
- each venue has deterministic caps (margin, liquidation distance, min/max order size)

**C) Exit prioritiser**
RL ranks candidate cross-venue exit chunks based on edge, basis impact, fragmentation reduction.

### 3) Observation Space (What RL Sees)
Minimum viable observation vector:
- global: fair value S_t, σ_t, vol_ratio, risk_regime, kill_switch, global inventory q_t, basis exposure B_t
- per-venue: mid, spread, depth, basis b_v, funding rate, toxicity, q_v, margin_available, dist_liq_sigma
- optional: latency proxies, recent fill markouts, venue health flags

### 4) Reward Design (Risk-Adjusted, Not Just PnL)
A safe default reward:
reward_t =
  + realised_pnl_t
  + w_funding * funding_pnl_t
  - w_fees * fees_t
  - w_slip * slippage_t
  - w_dd * drawdown_penalty_t
  - w_basis * |basis_exposure_t|
  - w_liq * liquidation_proximity_penalty_t
  - w_kill * I[kill_switch]

Prefer tail-aware shaping:
- penalise worst-quantile drawdowns (CVaR-style)
- penalise repeated near-limit behaviour

### 5) Training Stack (Most Advanced Practical Path)
**Stage 1 — Quant optimisation first (pre-RL)**
- Monte Carlo + adversarial stress + multi-objective tuning
- produces strong baselines and better simulators for RL

**Stage 2 — World model (learned simulator)**
- Train an ensemble dynamics model on telemetry and simulation rollouts
- Use uncertainty estimates (ensemble disagreement) for gating

**Stage 3 — Model-based RL on GPUs**
- Dreamer/MuZero-style latent world model RL for sample efficiency
- Constrained RL / Lagrangian methods for budgets
- Domain randomisation + adversarial scenario generation to prevent simulator overfitting

**Stage 4 — Offline RL bridge to production**
- Train on logged data + conservative offline RL objectives
- Validate on held-out scenarios and adversarial regression seeds

### 6) Deployment Plan (Shadow → Gated → Expanded)
1) Shadow mode
   - RL proposes actions, baseline executes
   - log divergences and counterfactual metrics

2) Gated execution
   - allow RL to control only a subset of knobs with strict caps
   - automatically revert to baseline if safety scorecard degrades

3) Expanded control
   - only after repeated success in out-of-sample + live shadow comparisons

### 7) Promotion Criteria (When RL Is Allowed More Control)
An RL policy revision is promoted only if it:
- improves tail metrics (CVaR / worst-quantile drawdown) out-of-sample,
- reduces kill probability with confidence bounds,
- passes adversarial regression suites,
- respects deterministic safety shield with zero violations,
- remains operationally stable (latency, throughput, no oscillatory behaviour).