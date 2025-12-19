# Paraphina Evidence Pack (v1)

Purpose: A hard reference map from the WHITEPAPER to concrete repo evidence (files + symbols).
Rule: We only claim what is directly supported by code and/or tests.

Conventions:
- Cite as: `path/to/file.rs:L12-L34` + `fn/type::name` + 1–3 sentence paraphrase of what the code does.
- If evidence cannot be found: write **UNCONFIRMED** and list what would confirm it (file/symbol/tests needed).

---

## 1) Core loop order (strategy → actions → execution)

- **Where:**
  - `paraphina/src/strategy.rs:L116-L244` + `fn StrategyRunner::run_simulation`
  
- **Evidence:**
  - The per-tick loop follows the spec order (L129-L189):
    1. **Engine tick** (L130-L131): `engine.seed_dummy_mids()` then `engine.main_tick()` updates FV/vol/toxicity/risk
    2. **MM intents** (L134-L148): `compute_mm_quotes()` → `mm_quotes_to_order_intents()` → gateway fills → recompute
    3. **Exit intents** (L151-L169): `exit::compute_exit_intents()` → gateway fills → recompute (runs BEFORE hedge)
    4. **Hedge intents** (L172-L189): `compute_hedge_plan()` → `hedge_plan_to_order_intents()` → gateway fills → recompute
  - Actions are emitted as `OrderIntent` structs (`types.rs:L30-L50`) with venue_index, side, price, size, purpose
  - Actions applied via `gateway.process_intents()` which returns `FillEvent` structs
  - After each fill batch, `state.recompute_after_fills()` updates inventory/basis/unrealised PnL

---

## 2) Market making (quoting / quote model)

- **Where:**
  - `paraphina/src/mm.rs:L139-L270` + `fn compute_mm_quotes`
  - `paraphina/src/mm.rs:L323-L647` + `fn compute_single_venue_quotes`

- **Evidence:**
  - **Reservation price terms** (L419-L441):
    - `r_v = S_t + β_b*b_v + β_f*f_v - γ*(σ_eff^2)*τ*( q_global - λ_inv*(q_v - q_target_v) )`
    - `basis_adj = beta_b * basis_v` (L432)
    - `funding_adj = beta_f * funding_pnl_per_unit` (L433)
    - `inv_term = gamma * sigma_eff^2 * tau * inv_deviation` (L438-L439)
  
  - **Spread model** (L378-L415):
    - AS half-spread: `δ* = (1/γ) * ln(1 + γ/k)` (L387)
    - Applies `spread_mult` from volatility (L390)
    - Minimum half-spread enforced: `edge_local_min + maker_cost + vol_buffer` (L393-L396)
    - Warning regime widens via `spread_warn_mult` (L402-L404)
    - Liquidation proximity widens up to +200% (L407-L411)
  
  - **Size model** (L499-L624):
    - Quadratic objective: `J(Q) = e*Q - 0.5*η*Q^2` → `Q_raw = e / η` (L503-L516)
    - Applies `size_mult` from volatility (L519-L520)
    - Margin cap: `Q_margin_max = margin * MM_MAX_LEVERAGE * MM_MARGIN_SAFETY / price` (L528-L540)
    - Liq-distance shrink: linear within (crit, warn) (L548-L568)
    - Delta-limit directional throttling: >2x limit stops quoting, 1-2x allows only risk-reducing (L587-L607)
  
  - **Inventory skew** (L435-L439):
    - `inv_deviation = q_global - lambda_inv * (q_v - q_target_v)`
    - Per-venue targets via `compute_venue_targets()` (L78-L137) using depth weights and funding preference
  
  - **Risk gating inputs** (L182-L196, L346-L375):
    - Kill switch OR HardLimit regime → return no quotes
    - Per-venue: Disabled status, liq < crit_sigma, toxicity ≥ tox_high → skip venue
    - Passivity enforced: bid ≤ best_bid - tick, ask ≥ best_ask + tick (L449-L472)

---

## 3) Exit engine (kill-switch / exits / unwind)

- **Where:**
  - `paraphina/src/exit.rs:L247-L600` + `fn compute_exit_intents`

- **Evidence:**
  - **Trigger conditions** (L257-L259):
    - Exit engine disabled when `kill_switch` is true OR `risk_regime == HardLimit`
    - Requires `|q_global| >= min_global_abs_tao` to act (L267-L269)
  
  - **Cancel/flatten semantics** (L282-L288, L533-L597):
    - Side determined by global position: `q > 0 → Sell`, `q < 0 → Buy` to reduce |q_global|
    - Greedy knapsack allocation across venues sorted by score
    - Profit-only gate: `base_profit_per_tao > edge_threshold` required (L405-L409)
  
  - **Hysteresis / reset behavior**:
    - No explicit hysteresis in exit engine; runs each tick if conditions met
    - Global/per-venue liq mode checked via `global_liq_mode()` (L145-L171) and `venue_liq_cap_mult()` (L177-L190)
  
  - **Reason reporting**:
    - Exit intents tagged with `OrderPurpose::Exit` (L593)
    - No explicit exit reason enum; exit actions are opportunistic profit-taking

---

## 4) Hedge allocator (global hedge)

- **Where:**
  - `paraphina/src/hedge.rs:L118-L196` + `fn compute_hedge_plan`
  - `paraphina/src/hedge.rs:L199-L377` + `fn build_candidates`
  - `paraphina/src/hedge.rs:L380-L427` + `fn greedy_allocate`

- **Evidence:**
  - **Deadband** (L135-L148):
    - `band_vol = band_base_tao * (1 + band_vol_mult * vol_ratio_clipped)`
    - If `|X| <= band_vol` then no hedge action (L146-L148)
  
  - **Allocation / cost model** (L298-L361):
    - Execution cost: `half_spread + taker_fee + slippage_buffer`
    - Funding benefit: directional based on side vs funding_8h
    - Basis edge: `raw_px - fair` (sell) or `fair - raw_px` (buy)
    - Liquidation penalty via `compute_liq_penalty()` (L95-L106)
    - Fragmentation penalty if opening new position on venue (L335-L339)
    - `total_cost = exec_cost - funding_weight * funding_benefit - basis_weight * basis_edge + liq_penalty + frag_penalty`
  
  - **Venue gating** (L213-L260):
    - Skip if `!vcfg.is_hedge_allowed`
    - Skip if Disabled status, stale book, toxicity ≥ tox_high, missing mid/spread
    - Skip if `depth_near_mid < min_depth_usd`
    - Hard-skip if `dist_liq_sigma <= liq_crit_sigma`

---

## 5) Fair value + volatility gating

- **Where:**
  - `paraphina/src/engine.rs:L99-L223` + `fn update_fair_value_and_vol`
  - `paraphina/src/engine.rs:L276-L348` + `fn collect_kf_observations`
  - `paraphina/src/engine.rs:L354-L408` + `fn update_vol_and_scalars`

- **Evidence:**
  - **Fair value definition / estimator** (L99-L223):
    - Robust multi-venue Kalman filter over log-price
    - Sequential measurement updates for each eligible venue
    - Uninitialised filter seeded from median mid across venues (L119, L226-L264)
  
  - **Outlier gating** (L321-L330):
    - Reject venues where `|mid - ref_price| / ref_price > max_mid_jump_pct`
  
  - **Vol estimator** (L354-L384):
    - Short/long EWMA of squared log returns
    - `sigma_short = sqrt(var_short)`, `sigma_long = sqrt(var_long)`
  
  - **Vol floor / sigma_eff** (L380-L384):
    - `sigma_eff = max(sigma_short, sigma_min)`
  
  - **FV gating telemetry** (L219-L222):
    - `fv_available`: true if ≥ `min_healthy_for_kf` observations
    - `healthy_venues_used`: indices of venues contributing to KF update

---

## 6) Toxicity + markout

- **Where:**
  - `paraphina/src/toxicity.rs:L40-L128` + `fn update_toxicity_and_health`
  - `paraphina/src/state.rs:L449-L474` + `fn record_pending_markout`

- **Evidence:**
  - **Markout horizon(s)** (config L362): `markout_horizon_ms` (default 5000ms = 5s)
  
  - **Computation (EWMA / thresholds)** (L76-L95):
    - Markout: `mid_now - fill_price` (buy) or `fill_price - mid_now` (sell)
    - If markout >= 0: `tox_instant = 0` (favorable)
    - If markout < 0: `tox_instant = clamp(-markout / markout_scale, 0, 1)`
    - EWMA: `tox = (1 - alpha) * tox + alpha * tox_instant`
  
  - **Venue disable rules** (L119-L127):
    - `tox >= tox_high_threshold` → `VenueStatus::Disabled`
    - `tox >= tox_med_threshold` → `VenueStatus::Warning`
    - else → `VenueStatus::Healthy`

---

## 7) Risk regime + kill switch semantics

- **Where:**
  - `paraphina/src/engine.rs:L421-L511` + `fn update_risk_limits_and_regime`
  - `paraphina/src/state.rs:L97-L124` + `enum RiskRegime`, `enum KillReason`

- **Evidence:**
  - **Regime definition(s)** (state.rs L98-L105):
    - `RiskRegime::Normal` / `RiskRegime::Warning` / `RiskRegime::HardLimit`
  
  - **Normal/Warning/HardLimit conditions** (engine.rs L464-L477):
    - **HardLimit** if ANY hard breach: `pnl_hard_breach || delta_hard_breach || basis_hard_breach || liq_hard_breach`
    - **Warning** if ANY warning threshold: `delta_abs >= delta_warn || basis_abs >= basis_warn || pnl <= pnl_warn || dist_liq <= liq_warn_sigma`
    - **Normal** otherwise
  
  - **KillReason / reporting** (state.rs L107-L124):
    - `KillReason::None | PnlHardBreach | DeltaHardBreach | BasisHardBreach | LiquidationDistanceBreach`
    - Kill switch latched once triggered (engine.rs L488-L503): checked in priority order PnL→Delta→Basis→Liq
    - Once `kill_switch = true`, regime forced to HardLimit (L506-L510)
    - `print_daily_summary()` outputs kill reason (strategy.rs L285-L287)

---

## 8) Research harness + metrics

- **Where:**
  - `batch_runs/orchestrator.py` + `batch_runs/metrics.py`
  - `batch_runs/exp03_stress_search.py` (template)

- **Evidence:**
  - **Determinism controls**:
    - Config loaded via `Config::from_env_or_default()` (config.rs L959-L982)
    - Seed passed via `StrategyRunner::set_seed()` (strategy.rs L68-L70)
    - Timebase offset: `base_ms = seed % 10_000` (strategy.rs L119)
  
  - **Metrics definitions** (metrics.py L50-L88):
    - `parse_daily_summary()` extracts: `pnl_realised`, `pnl_unrealised`, `pnl_total`, `kill_switch`
    - Regex patterns match Rust stdout format (metrics.py L35-L47)
    - `aggregate_basic()` computes: mean/std of PnL, `kill_switch_frac`, `num_runs`
  
  - **Stress harness expectations**:
    - exp03 template: load prior results → derive parameter centre → build grid of env overlays → `run_many()` → write CSVs
    - Stdout parser contract: lines like `Daily PnL (realised / unrealised / total): +X / +Y / +Z`
    - Kill switch format: `Kill switch: true|false`

---

## 9) RL / training scaffolding

- **Where:**
  - `paraphina/src/rl/` (schemas, env, trajectory)
  - `python/rl2_bc/` (dataset/train/eval)

- **Evidence:**
  - **Observation schema + versioning** (`rl/observation.rs`):
    - `const OBS_VERSION: u32 = 1` (L22)
    - `struct Observation` (L141-L210): global features (FV, vol, inventory, basis, PnL, risk) + per-venue features
    - `struct VenueObservation` (L28-L70): mid, spread, depth, vol, status, toxicity, position, funding, margin, liq-distance
    - Constructed via `Observation::from_state()` (L217-L274)
  
  - **Action encoding + versioning** (`rl/action_encoding.rs`):
    - `const ACTION_VERSION: u32 = 1` (L21)
    - Per-venue: `spread_scale [0.5,3.0]`, `size_scale [0.0,2.0]`, `rprice_offset_usd [-10,10]` (bounds L64-L84)
    - Global: `hedge_scale [0.0,2.0]`, `hedge_venue_weights` (simplex)
    - `encode_action()` / `decode_action()` with normalization (L126-L259)
    - Round-trip tests verify determinism (L452-L463)
  
  - **Trajectory recording** (`rl/trajectory.rs`):
    - `const TRAJECTORY_VERSION: u32 = 1` (L27)
    - `TrajectoryRecord`: obs_features, action_target, reward, terminal, kill_reason, episode/step indices (L132-L150)
    - `TrajectoryCollector::collect()` runs vectorized rollouts (L226-L405)
    - `TrajectoryWriter` outputs to JSONL or binary arrays (L418-L605)
  
  - **Dataset generation/train/eval scripts** (`python/rl2_bc/`):
    - `generate_dataset.py`: collects trajectories from HeuristicPolicy
    - `train_bc.py`: trains MLP to imitate heuristic
    - `eval_bc.py`: evaluates action prediction error and rollout performance
    - Exit criteria (README.md L177-L183): MAE < 0.1, kill rate ≤ 1.1×, PnL ≥ 0.9×

---

## 10) Known gaps (explicit)

- **Unimplemented or stubbed behavior:**
  - `rl/runner.rs:L368` TODO: "Apply policy_action modifiers to mm_intents" — policy action not yet wired to MM quotes
  - `rl/runner.rs:L398` TODO: "Apply policy_action.hedge_scale and hedge_venue_weights" — policy action not yet wired to hedge allocation
  - RL shadow runner exists but policy actions are identity passthrough; learned policy integration pending

- **Assumptions that are not yet modeled:**
  - Synthetic mids/spreads/depth in simulation (`engine.rs:L33-L71`) — not connected to real L2 data
  - Margin/liquidation fields initialized to synthetic defaults (`state.rs:L230-L235`)
  - No real order execution; fills are simulated via gateway abstraction

- **"Whitepaper claims that currently exceed evidence" (if any):**
  - Whitepaper describes full RL control surface (spread_scale, size_scale, hedge weights) — implementation scaffolded but TODOs remain for actual application
  - Model-based RL / world model training mentioned in playbook — scripts exist (`wm01_train_world_model.py` etc.) but integration with main strategy pending
  - ONNX export for deployment mentioned in playbook — not yet implemented

---

## Test coverage summary

- **Unit tests found:** 102 `#[test]` functions across 19 source files
- **Key test coverage:**
  - `mm.rs`: passivity, reservation price monotonicity, HardLimit/kill-switch produce no quotes, toxicity gating, venue targets (L867-L136)
  - `exit.rs`: lot size rounding, min notional, slippage buffer, fragmentation scoring (L602-L684)
  - `hedge.rs`: lot size rounding, top-of-book derivation, liq penalty (L464-L527)
  - `toxicity.rs`: favorable/adverse markouts, horizon timing, venue disable (L130-L333)
  - `state.rs`: basis/PnL calculations, fill application (L502-L625)
  - `rl/observation.rs`: serialization determinism, roundtrip (L311-L416)
  - `rl/action_encoding.rs`: encode/decode roundtrip, clamping, normalization (L286-L543)
  - `rl/trajectory.rs`: obs encoding, collector determinism (L608-L698)

