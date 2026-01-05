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

### Milestone A — Documentation + "contracts first"
**Goal:** make it impossible to confuse "planned spec" with "implemented behavior".

**Status: COMPLETE**

Deliverables:
- `docs/WHITEPAPER.md` is the single entry point:
  - "Implementation truth" section stays conservative and code-backed
  - "Canonical spec" is preserved verbatim
- Add/maintain a short "Drift Register" section listing:
  - spec features not implemented yet,
  - known mismatches in naming/semantics (e.g., Warning/Critical vs Warning/HardLimit).

**What shipped:**
- `docs/WHITEPAPER.md`: Two-part structure with implementation truth (Part I) and canonical spec (Part II)
- `docs/EVIDENCE_PACK.md`: Hard reference map from WHITEPAPER to code evidence
- `docs/AI_PLAYBOOK.md`: Development guidelines
- Docs Integrity Gate (Implemented refs + canonical spec hash lock) — COMPLETE

**Evidence:** See `docs/EVIDENCE_PACK.md` §1 (Core loop), §8 (Research harness)

**Acceptance criteria:**
- Invariant: `docs/WHITEPAPER.md` contains "Known drift" section listing spec vs implementation gaps
- Invariant: Every algorithmic claim in Part I has `Implemented:` annotation with file path
- Manual: A new contributor can run `exp03` and understand env knobs + stdout format

---

### Milestone B — Standardise knobs and experimentability
**Goal:** every strategy parameter that matters is:
- configurable (env/config),
- logged/observable,
- sweepable via orchestrator labels.

**Status: COMPLETE**

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

**What shipped:**
- `paraphina/src/config.rs`: `Config::from_env_or_profile()`, `Config::from_env_or_default()` with all env var overrides
- `batch_runs/metrics.py`: `parse_daily_summary()` regex parser for stdout
- `batch_runs/orchestrator.py`: `EngineRunConfig`, `run_many()`, `results_to_dataframe()`

**Evidence:** See `docs/EVIDENCE_PACK.md` §8 (Research harness + metrics)

**Acceptance criteria:**
- Passes `paraphina/tests/config_profile_tests.rs` (profile env override tests)
- Passes `paraphina/tests/metrics_tests.rs` (stdout parsing tests)
- Determinism: `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_single_tick` passes with fixed seed

---

### Milestone C — Risk regime correctness + kill switch semantics
**Goal:** risk states match the canonical model: Normal / Warning / Critical
with a kill switch that is explicit and unambiguous.

**Status: COMPLETE**

Deliverables:
- Define regime semantics and thresholds in one place.
- Ensure **Critical implies kill_switch** behaviorally:
  - cancel/stop new orders,
  - allow only bounded "best-effort" risk reduction if implemented.

**What shipped:**
- `paraphina/src/state.rs`: `RiskRegime` enum (Normal/Warning/HardLimit), `KillReason` enum
- `paraphina/src/engine.rs::update_risk_limits_and_regime()`: Latching kill switch logic (L488-L510)
- `paraphina/src/mm.rs::compute_mm_quotes()`: Returns no quotes when `kill_switch || risk_regime == HardLimit`
- `paraphina/src/hedge.rs::compute_hedge_orders()`: Returns empty when `kill_switch`
- `paraphina/src/exit.rs::compute_exit_intents()`: Returns empty when `kill_switch`

**Evidence:** See `docs/EVIDENCE_PACK.md` §7 (Risk regime + kill switch semantics)

**Acceptance criteria:**
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_and_kill_switch_when_loss_limit_breached`
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_and_kill_switch_when_delta_limit_breached`
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_and_kill_switch_when_basis_limit_breached`
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_and_kill_switch_when_liquidation_distance_breached`
- Passes `paraphina/tests/risk_regim_tests.rs::kill_switch_latches_once_true`
- Passes `paraphina/tests/risk_regim_tests.rs::hardlimit_from_delta_breach_triggers_kill_and_disables_mm`
- Passes `paraphina/tests/risk_regim_tests.rs::hedge_disabled_when_kill_switch_active`
- Passes `paraphina/tests/risk_regim_tests.rs::exit_disabled_when_kill_switch_active`
- Invariant: `state.kill_switch == true` implies `state.risk_regime == HardLimit`
- Invariant: `KillReason` preserved from first breach (latching)

---

### Milestone D — Fair value + volatility gating (robustness)
**Goal:** fair value and volatility estimates are stable, and strategy behavior
degrades safely when data quality drops.

**Status: COMPLETE**

Deliverables:
- FV estimator gating policy (stale books / outliers / min healthy venues).
- Vol floor behavior documented and tested.
- Telemetry includes:
  - fair value (or a "fv_available" flag),
  - effective volatility,
  - list/count of healthy venues used.

**What shipped:**
- `paraphina/src/engine.rs::update_fair_value_and_vol()`: Kalman filter with venue gating (L99-L223)
- `paraphina/src/engine.rs::collect_kf_observations()`: Staleness/outlier/min-healthy gating (L276-L348)
- `paraphina/src/engine.rs::update_vol_and_scalars()`: EWMA vol + sigma_eff floor (L354-L408)
- Telemetry fields: `fv_available`, `healthy_venues_used`, `healthy_venues_used_count`, `sigma_eff`

**Evidence:** See `docs/EVIDENCE_PACK.md` §5 (Fair value + volatility gating)

**Acceptance criteria:**
- Passes `paraphina/tests/fair_value_gating_tests.rs::stale_venue_data_excluded_from_fv_update`
- Passes `paraphina/tests/fair_value_gating_tests.rs::fv_degrades_gracefully_with_all_stale_data`
- Passes `paraphina/tests/fair_value_gating_tests.rs::outlier_venue_excluded_from_fv_update`
- Passes `paraphina/tests/fair_value_gating_tests.rs::min_healthy_threshold_enforced`
- Passes `paraphina/tests/fair_value_gating_tests.rs::telemetry_fields_populated_correctly`
- Passes `paraphina/tests/vol_floor_tests.rs::sigma_eff_never_below_sigma_min`
- Passes `paraphina/tests/vol_floor_tests.rs::sigma_eff_uses_floor_when_raw_vol_is_low`
- Passes `paraphina/tests/vol_floor_tests.rs::vol_scalars_use_sigma_eff`
- Invariant: `sigma_eff = max(fv_short_vol, sigma_min)`
- Invariant: `fv_available = false` when `healthy_venues_used_count < min_healthy_for_kf`

---

### Milestone E — Cross-venue exits (canonical Section 12)
**Goal:** implement cross-perp exit optimization that:
- uses net edge after fees/slippage/vol buffers,
- incorporates basis/funding differentials,
- penalises basis risk increase and fragmentation.

**Status: COMPLETE**

Deliverables:
- A discrete "exit allocator" module callable after fill batches.
- A consistent definition of:
  - fragmentation score,
  - basis exposure change approximation,
  - effective edge threshold(s).

**What shipped:**
- `paraphina/src/exit.rs::compute_exit_intents()`: Full exit allocator (L247-L600)
- Edge calculation: `base_profit - fees - slippage_buffer - vol_buffer + basis_adj + funding_adj`
- Fragmentation penalty: `fragmentation_penalty_per_tao` for opening new legs
- Fragmentation bonus: `fragmentation_reduction_bonus` for closing positions
- Basis-risk penalty: `basis_risk_penalty_weight * Δ|B_t|`
- Per-venue constraints: `lot_size_tao`, `size_step_tao`, `min_notional_usd`

**Evidence:** See `docs/EVIDENCE_PACK.md` §3 (Exit engine)

**Acceptance criteria:**
- Passes `paraphina/tests/exit_engine_tests.rs::exit_respects_lot_size_and_min_notional`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_respects_min_notional`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_profit_only_blocks_unprofitable_exits`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_splits_across_best_venues_when_capped_per_venue`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_skips_disabled_or_toxic_or_stale`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_prefers_less_fragmentation_when_edges_similar`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_prefers_less_basis_risk_when_edges_similar`
- Passes `paraphina/tests/exit_engine_tests.rs::exit_deterministic_ordering_with_identical_edges`
- Invariant: Exit only when `base_profit_per_tao > edge_min_usd`
- Invariant: Exit size rounded to `size_step_tao` and >= `lot_size_tao`
- Determinism: Same state produces identical exit intents

---

### Milestone F — Global hedge allocator (canonical Section 13)
<!-- STATUS: MILESTONE_F = COMPLETE -->
**Goal:** hedging is a *global* optimization over allowed venues:
- LQ controller decides global step size with deadband,
- allocation chooses cheapest/least-risk venues first,
- constraints include funding/basis/margin/liquidation/fragmentation.

**Status: COMPLETE**

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

**What shipped:**
- `paraphina/src/hedge.rs::compute_hedge_plan()`: Global hedge with deadband (L118-L196)
- `paraphina/src/hedge.rs::build_candidates()`: Per-venue cost model (L199-L377)
- `paraphina/src/hedge.rs::greedy_allocate()`: Greedy allocation by cost (L380-L427)
- Cost components: `exec_cost + liq_penalty + frag_penalty - funding_benefit - basis_edge`

**Evidence:** See `docs/EVIDENCE_PACK.md` §4 (Hedge allocator)

**Acceptance criteria:**
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_deadband_no_orders`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_outside_deadband_generates_orders`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_respects_global_max_step`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_respects_per_venue_caps`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_avoids_near_liquidation`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_skips_critical_liquidation_venues`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_prefers_funding_or_basis_when_exec_equal`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_prefers_basis_when_funding_equal`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_deterministic_tiebreak_by_venue_index`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_skips_disabled_stale_toxic_venues`
- Passes `paraphina/tests/hedge_allocator_tests.rs::hedge_disabled_when_kill_switch_active`
- Invariant: `|X| <= band_vol` implies no hedge orders
- Invariant: Total hedge size <= `max_step_tao`
- Invariant: Venues at `dist_liq_sigma <= liq_crit_sigma` are hard-skipped

---

### Milestone G — Multi-venue quoting model (canonical Section 9–11)
**Goal:** quoting per venue reflects:
- Avellaneda–Stoikov baseline,
- basis/funding shifts,
- per-venue inventory targets,
- toxicity and liquidation-distance-aware size constraints,
- stable order management (min quote lifetime, tolerance thresholds).

**Status: COMPLETE**

Deliverables:
- Explicit functions with unit tests for:
  - reservation price components
  - half-spread computation
  - edge filters
  - size constraints and shrink logic
  - cancel/replace logic thresholds

**What shipped:**
- `paraphina/src/mm.rs::compute_mm_quotes()`: Full quote generation (L139-L270)
- `paraphina/src/mm.rs::compute_single_venue_quotes()`: Per-venue AS model (L323-L647)
- Reservation price: `r_v = S_t + β_b*b_v + β_f*f_v - γ*(σ_eff^2)*τ*inv_deviation` (L419-L441)
- AS half-spread: `δ* = (1/γ) * ln(1 + γ/k)` with vol scaling (L378-L415)
- Size model: Quadratic objective with margin/liq-distance/delta-limit constraints (L499-L624)
- Order management: `should_replace_order()` with lifetime and tolerance logic
- Per-venue targets: `compute_venue_targets()` with depth and funding weights (L78-L137)

**Evidence:** See `docs/EVIDENCE_PACK.md` §2 (Market making / quote model)

**Acceptance criteria:**
- Passes `paraphina/tests/mm_quote_model_tests.rs::passivity_bid_strictly_below_best_bid`
- Passes `paraphina/tests/mm_quote_model_tests.rs::passivity_ask_strictly_above_best_ask`
- Passes `paraphina/tests/mm_quote_model_tests.rs::passivity_bid_ask_do_not_cross`
- Passes `paraphina/tests/mm_quote_model_tests.rs::reservation_price_decreases_when_long`
- Passes `paraphina/tests/mm_quote_model_tests.rs::reservation_price_increases_when_short`
- Passes `paraphina/tests/mm_quote_model_tests.rs::hardlimit_produces_no_quotes_even_if_kill_switch_false`
- Passes `paraphina/tests/mm_quote_model_tests.rs::kill_switch_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::warning_regime_widens_spread`
- Passes `paraphina/tests/mm_quote_model_tests.rs::warning_regime_caps_size`
- Passes `paraphina/tests/mm_quote_model_tests.rs::disabled_venue_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::high_toxicity_venue_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::critical_liquidation_distance_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::size_respects_max_order_size`
- Passes `paraphina/tests/mm_quote_model_tests.rs::size_respects_lot_size`
- Passes `paraphina/tests/mm_quote_model_tests.rs::liquidation_warning_shrinks_size`
- Passes `paraphina/tests/mm_quote_model_tests.rs::venue_targets_scale_with_depth`
- Passes `paraphina/tests/mm_quote_model_tests.rs::young_passive_order_not_replaced`
- Passes `paraphina/tests/mm_quote_model_tests.rs::old_order_with_price_change_replaced`
- Passes `paraphina/tests/mm_quote_model_tests.rs::extreme_delta_produces_no_quotes`
- Passes `paraphina/tests/mm_quote_model_tests.rs::high_delta_only_allows_risk_reducing`
- Invariant: `bid < best_bid - tick` and `ask > best_ask + tick` (passivity)
- Invariant: `q_global > 0` decreases reservation price (skew toward selling)
- Invariant: `RiskRegime::HardLimit || kill_switch` implies no quotes

---

### Milestone H — Production-ready separation (I/O vs strategy)
**Goal:** preserve a clean boundary:
- strategy core is pure/deterministic
- I/O layer (WS/REST) is async and replaceable
- execution gateway handles venue semantics (TIF/post-only/IOC guard).

**Status: COMPLETE**

Deliverables:
- Action interface: `PlaceOrder`, `CancelOrder`, `CancelAll`, `SetKillSwitch`, …
- Venue adapter trait per exchange
- Rate limiting and retry policies (configurable)

**What shipped:**
- `paraphina/src/actions.rs`: `Action` enum with `PlaceOrder`/`CancelOrder`/`CancelAll`/`SetKillSwitch`
- `paraphina/src/io/mod.rs`: `IoAdapter` trait for venue communication
- `paraphina/src/io/sim.rs`: `SimulatedIoAdapter` for deterministic replay
- `paraphina/src/io/noop.rs`: `NoopIoAdapter` for testing
- `paraphina/src/strategy_core.rs`: Pure strategy layer, no I/O side effects
- `paraphina/src/strategy.rs::StrategyRunner`: Separated event ingestion from action generation

**Evidence:** See `docs/EVIDENCE_PACK.md` §1 (Core loop), §6 (RL / policy interface)

**Acceptance criteria:**
- Passes `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_single_tick`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_multi_tick`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_with_inventory`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_action_ids_deterministic`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_different_seeds_produce_different_batches`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_kill_switch_determinism`
- Passes `paraphina/tests/replay_determinism_tests.rs::test_strategy_output_intents_match_actions`
- Determinism: `paraphina/tests/replay_determinism_tests.rs::test_replay_determinism_multi_tick` passes with fixed seed
- Invariant: `IoAdapter` trait has no mutable access to `GlobalState`
- Invariant: Same `(initial_state, event_stream, config, seed)` produces identical action stream

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
**Goal:** make the current engine "RL-ready" without changing behaviour.

**Status: COMPLETE**

- [x] Add a versioned `Observation` schema derived from `GlobalState` + venues
- [x] Add a `Policy` interface with a default `HeuristicPolicy` (current logic)
- [x] Log policy inputs/outputs in telemetry (obs_version, policy_version)
- [x] Add deterministic seeding + episode reset mechanics in simulation mode
- [x] Add a "shadow mode" runner (policy proposes, baseline executes)

**What shipped:**
- `src/rl/observation.rs`: `Observation`, `VenueObservation` with `OBS_VERSION=1`
- `src/rl/policy.rs`: `Policy` trait, `HeuristicPolicy` (identity pass-through)
- `src/rl/telemetry.rs`: Policy input/output logging
- `src/rl/runner.rs`: `ShadowRunner` for policy proposal without execution
- `src/rl/sim_env.rs`: `SimEnv` with deterministic seeding and episode reset

**Evidence:** See `docs/EVIDENCE_PACK.md` §9 (RL / training scaffolding)

**Acceptance criteria:**
- Passes `paraphina/tests/rl_determinism_tests.rs::test_observation_deterministic`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_observation_version`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_policy_action_deterministic`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_episode_determinism`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_different_seeds_produce_different_results`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_shadow_mode_no_execution_impact`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_heuristic_policy_is_identity`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_policy_reset_episode`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_multiple_episodes_with_reset`
- Determinism: Same seed produces identical observation sequences
- Invariant: HeuristicPolicy returns identity action (no modifications)

**Exit criteria: SATISFIED**
- Replays are deterministic and byte-for-byte reproducible ✓
- Telemetry is sufficient to reconstruct rewards and constraints ✓

### RL-1: Gym-style environment and vectorised simulation
**Goal:** turn the simulator into a high-throughput training environment.

**Status: COMPLETE**

- [x] Create `paraphina_env` wrapper (Python bindings) around Rust sim
- [x] Support vectorised rollouts (N environments in parallel)
- [x] Add domain randomisation knobs:
  - fees, spreads, slippage model
  - funding regimes
  - volatility regimes
  - venue staleness / disable events

**What shipped:**
- `src/rl/sim_env.rs`: `SimEnv` with step/reset/obs interface
- `src/rl/sim_env.rs`: `VecEnv` for N parallel environments
- `src/rl/domain_rand.rs`: `DomainRandSampler` with fee/spread/funding/volatility randomisation
- `paraphina_env`: PyO3 bindings exposing `SimEnv`, `VecEnv`, `TrajectoryCollector`

**Evidence:** See `docs/EVIDENCE_PACK.md` §9 (RL / training scaffolding)

**Acceptance criteria:**
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_sim_env_determinism_same_seed_same_actions`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_sim_env_determinism_with_domain_rand`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_sim_env_different_seeds_different_results`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_vec_env_determinism`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_vec_env_smoke`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_vec_env_custom_actions`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_domain_rand_sampler_determinism`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_episode_termination_max_ticks`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_episode_termination_kill_switch`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_observation_structure`
- Passes `paraphina/tests/rl_env_determinism_tests.rs::test_vec_env_independence`
- Invariant: VecEnv episodes are independent (no cross-contamination)
- Invariant: Same seed + domain_rand_seed produces identical rollouts

**Exit criteria: SATISFIED**
- Sustained high-step throughput (enough to saturate GPU training) ✓
- Training/eval parity: same reward + constraints computed everywhere ✓

### RL-2: Imitation learning baseline (behaviour cloning)
**Goal:** train a policy to imitate the heuristic strategy.

**Status: COMPLETE**

- [x] Generate large trajectory datasets from heuristic policy
- [x] Train BC policy on bounded "control surface" actions (spread/size/offset)
- [x] Evaluate: action error, PnL parity, risk parity under randomisation

**Exit criteria**
- BC policy matches baseline within tolerance and does not increase kill rate

**What shipped:**
- `src/rl/action_encoding.rs`: Versioned action encoding (`ACTION_VERSION=1`) with bounded ranges
- `src/rl/trajectory.rs`: `TrajectoryCollector` for deterministic dataset generation
- `src/rl/observation.rs`: Versioned observation schema (`OBS_VERSION=1`)
- `paraphina_env`: Python bindings for `TrajectoryCollector` class
- `python/rl2_bc/`: Dataset generation, BC training (PyTorch MLP), and evaluation scripts

**Evidence:** See `docs/EVIDENCE_PACK.md` §9 (RL / training scaffolding)

**Acceptance criteria:**
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_determinism_identity`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_determinism_varied`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_round_trip_identity`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_round_trip_extreme_values`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_action_encoding_clamping`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_trajectory_collection_determinism_small`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_trajectory_collection_different_seeds`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_trajectory_metadata_versions`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_trajectory_record_dimensions`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_observation_to_features_determinism`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_heuristic_policy_determinism`
- Passes `paraphina/tests/rl2_bc_tests.rs::test_heuristic_policy_produces_identity`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_observation_deterministic`
- Passes `paraphina/tests/rl_determinism_tests.rs::test_episode_determinism`
- Invariant: `ACTION_VERSION` and `OBS_VERSION` match between train and inference
- Invariant: Same seed produces byte-identical trajectories

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

**Status: COMPLETE (A1 + A2 fully implemented)**

- [x] **Scenario library (seeded, reproducible)** (v1 COMPLETE, promotion-critical)
  - **v1 implemented (10 scenarios):**
    - Volatility regimes: low/medium/high (3)
    - Liquidity shocks: spread widening + depth thinning (2)
    - Venue outage: disabled venue window (1)
    - Funding inversions: sign flip + drift (2)
    - Basis spikes: positive + negative (2)
  - **Implemented:** `batch_runs/phase_a/scenario_library_v1.py` (generator/check/smoke CLI)
  - **Manifest:** `scenarios/v1/scenario_library_v1/manifest_sha256.json` (SHA-256 verified)
  - **Smoke suite:** `scenarios/suites/scenario_library_smoke_v1.yaml` (CI-friendly, 5 scenarios)
  - **Full suite:** `scenarios/suites/scenario_library_v1.yaml` (all 10 scenarios, promotion-critical)
  - **CI workflow:** `.github/workflows/scenario_library_smoke.yml`
  - **Integrated into Phase A Promotion Pipeline:**
    - `batch_runs/phase_a/promote_pipeline.py` includes scenario library by default
    - In `--smoke` mode: uses `scenario_library_smoke_v1.yaml` (5 scenarios)
    - In full mode: uses `scenario_library_v1.yaml` (all 10 scenarios)
    - CLI flags: `--skip-scenario-library`, `--scenario-library-suite PATH`
    - Manifest integrity verified at pipeline start (fail-fast if mismatch)
    - Results recorded in `PROMOTION_RECORD.json` (ran/skipped/passed/errors)
    - Evidence verification includes scenario library suite artifacts
  - **Remaining for v2:** latency / partial fill / cancel storm modelling
- [x] **Tail risk metrics emitted** (A1)
  - `mc_summary.json` schema_version=2 includes `tail_risk` section
  - PnL quantiles (p01, p05, p50, p95, p99)
  - PnL VaR/CVaR at alpha=0.95
  - Max drawdown quantiles and VaR/CVaR
  - Kill probability with Wilson 95% CI (point estimate, lower, upper)
  - **Implemented:** `paraphina/src/tail_risk.rs`, `paraphina/src/bin/monte_carlo.rs`
- [x] **Pareto harness scaffold** (A1)
  - `batch_runs/exp_phase_a_pareto_mc.py` provides:
    - Deterministic knob sweeps (grid or seeded random)
    - Isolated candidate runs with evidence pack verification
    - Pareto frontier computation (multi-objective)
    - Risk-tier budget selection (kill_prob_ci_upper, drawdown_cvar, min_mean_pnl)
    - Promoted config output (env file + promotion record JSON)
  - Usage: `python3 batch_runs/exp_phase_a_pareto_mc.py --smoke`
- [ ] Monte Carlo runner at scale
  - generate thousands–millions of scenarios
  - (foundation exists via monte_carlo binary; need advanced scenario generation)
- [x] **Adversarial / worst-case search** (A2) **IMPLEMENTED**
<!-- STATUS: CEM = IMPLEMENTED -->
  - `batch_runs/phase_a/adversarial_search_promote.py` provides:
    - **Cross-Entropy Method (CEM)** adversarial search (per WHITEPAPER B2)
    - Maintains mean/std per continuous parameter with elite fraction update
    - Deterministic scenario generation with stable filenames
    - Adversarial scoring: maximize kill_switch, drawdown; minimize mean_pnl
    - Top-K failure scenario promotion to `scenarios/v1/adversarial/generated_v1/`
    - Path-based regression suite: `scenarios/suites/adversarial_regression_v2.yaml`
    - Uses `sim_eval run` with `--output-dir` for isolated outputs
    - Verifies evidence with `verify-evidence-tree`
    - Python unit tests: `batch_runs/phase_a/tests/test_adversarial_search.py`
  - Legacy v1 harness: `batch_runs/exp_phase_a_adversarial_search.py`
    - Generates v1 suite with inline env_overrides
  - CI gate: `.github/workflows/adversarial_regression.yml`
    - Runs adversarial smoke search + suite on PRs
    - Verifies evidence packs for all scenarios
  - Usage: `python3 -m batch_runs.phase_a.adversarial_search_promote --smoke --out runs/adv_smoke`
  - Documentation: `docs/PHASE_A_ADVERSARIAL_SEARCH.md`
  - **Remaining:** ADR integration, time-to-failure minimization
- [x] **Multi-objective tuning of strategy knobs** (A2)
  - `batch_runs/phase_a/promote_pipeline.py` provides:
    - Deterministic candidate generation (seeded RNG + evolutionary mutation)
    - Multi-objective Pareto frontier computation
    - Objectives: maximize mean_pnl, minimize kill_prob_ci_upper, minimize drawdown_cvar
    - Budget-tier selection with deterministic tie-breaking
    - Outputs: `trials.jsonl`, `pareto.json`, `pareto.csv`
  - Usage: `python3 -m batch_runs.phase_a.promote_pipeline --smoke --study-dir runs/phaseA_smoke`
  - Unit tests: `batch_runs/phase_a/tests/test_pareto.py`, `test_budgets.py`, `test_winner_selection.py`
- [x] **Promotion pipeline** (A2)
  - `batch_runs/phase_a/promote_pipeline.py` implements budget-gated promotion:
    - Creates isolated trial directories: `runs/phaseA/<study>/<trial_id>/`
    - Writes `candidate.env` with configuration overrides
    - Runs `monte_carlo` with evidence pack generation
    - Runs out-of-sample suite: `scenarios/suites/research_v1.yaml`
    - Runs adversarial regression suite: `scenarios/suites/adversarial_regression_v1.yaml`
    - Verifies evidence packs (`sim_eval verify-evidence-tree`)
    - Parses metrics from `mc_summary.json` (JSON artifacts, not stdout)
    - Promotes winners to: `configs/presets/promoted/<tier>/phaseA_<study>_<timestamp>.env`
    - Writes `PROMOTION_RECORD.json` with full provenance
  - Documentation: `docs/PHASE_A_PROMOTION_PIPELINE.md`
  - Unit tests: `batch_runs/phase_a/tests/test_env_parsing.py`
- [x] **Phase AB Promotion Gate (Strict Mode)** **IMPLEMENTED**
  - `batch_runs/phase_ab/cli.py` provides the `gate` command:
    - Institutional-grade promotion gate with deterministic exit codes
    - Exit codes: PASS=0, FAIL=1, HOLD=2, ERROR=3 (distinct and auditable)
    - Mandatory evidence verification (cannot be skipped)
    - Required seed for reproducibility
    - Writes all standard outputs: `phase_ab_manifest.json`, `confidence_report.json`, `confidence_report.md`, `evidence_pack/manifest.json`, `phase_ab_summary.json`
  - CLI: `python3 -m batch_runs.phase_ab.cli gate --out-dir <path> --seed <int> [--auto-generate-phasea | --candidate-run <path>]`
  - CI workflow: `.github/workflows/phase_ab_promotion_gate.yml`
    - Manual dispatch (workflow_dispatch) for controlled promotion decisions
    - Configurable seed and n_bootstrap via workflow inputs
    - Uploads artifacts: `phase-ab-gate-artifacts`, `phase-ab-gate-evidence-pack`
    - Writes detailed GitHub Actions job summary
  - Unit tests: `tests/test_phase_ab_exit_codes.py` (32 tests covering exit code contracts)
  - Documentation: `docs/PHASE_AB_PIPELINE.md`
  - **Exit code contract (strict mode):**
    - 0 = PASS (PROMOTE - candidate is provably better)
    - 1 = FAIL (REJECT - candidate fails guardrails)
    - 2 = HOLD (insufficient evidence - needs more data)
    - 3 = ERROR (runtime/IO/parsing failure)

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