# Simulation & Evaluation Spec (v1)

Goal: make simulation runs reproducible, comparable, and CI-friendly.
Rule: no “results” claims unless they are produced by a committed scenario + seed + commit hash.

---

## 1) Scenario Contract (inputs)

A scenario is a versioned spec that fully defines a run:
- scenario_id (string)
- scenario_version (integer)  # start at 1
- engine (enum): { rl_sim_env | core_sim | replay_stub }
- horizon:
  - steps (int) OR duration_seconds (float)
  - dt_seconds (float)
- rng:
  - base_seed (u64)
  - num_seeds (int)  # runner expands base_seed + k
- initial_state:
  - risk_profile (string)  # uses existing PARAPHINA_RISK_PROFILE values
  - init_q_tao (float)
- market_model:
  - type: { synthetic | historical_stub }
  - synthetic (if synthetic):
      - process: { gbm | jump_diffusion_stub }
      - params: (vol, drift, jump_intensity, jump_size, etc)
  - historical_stub:
      - dataset_id (string)  # no data in repo; pointer only
- microstructure_model (v1 minimal):
  - fees_bps_maker (float)
  - fees_bps_taker (float)
  - latency_ms (float)  # constant in v1
- invariants (assertions):
  - expect_kill_switch: { always | never | allowed }
  - pnl_linearity_check: { enabled | disabled }  # for q0 sweeps, if applicable

---

## 2) Output Schema (artifacts)

Runner outputs a directory per run:
runs/<scenario_id>/<timestamp_or_commit>/<seed_k>/

Required files:
- run_summary.json  # small, stable; CI compares this
- metrics.jsonl     # optional streaming metrics (tick or periodic)
- config_resolved.json  # fully resolved scenario + defaults applied
- build_info.json   # git sha, dirty flag, rustc/cargo version

Minimum required fields in run_summary.json:
- scenario_id, scenario_version
- seed
- risk_profile, init_q_tao
- steps, dt_seconds
- pnl:
  - final_pnl_usd
  - max_drawdown_usd
- kill_switch:
  - triggered (bool)
  - step (int|null)
  - reason (string|null)
- determinism:
  - checksum (string)  # hash of key time series or summary payload

---

## 3) Evaluation Protocol (how changes are judged)

We maintain:
- A “CI smoke suite” scenario set (small, fast).
- A “research suite” scenario set (larger, used locally/overnight).

CI gates (v1):
- Determinism: identical `(scenario_id, seed)` => identical `run_summary.json` checksum.
- Schema completeness: required output fields always present.
- No regression in invariants:
  - if scenario says expect_kill_switch=never, kill-switch must not trigger.
  - if pnl_linearity_check enabled, it must pass within tolerance.

---

## 4) Calibration Targets (documented, not enforced in v1)

For synthetic processes:
- return mean ~ 0 (unless specified)
- realized vol matches requested target within tolerance
- jump frequency ~ intensity target (if jump model used)

For microstructure:
- fee sign flip should change performance directionally (maker rebates vs costs)
- no NaNs; no negative inventories if bounded

---

## 5) Ablation Harness (v1 registry)

Ablations are named switches the runner can apply:
- disable_toxicity_gate
- disable_fair_value_gate
- disable_vol_floor
- disable_risk_regime
- disable_exit_engine (should be forbidden in CI unless explicitly allowed)

Each ablation must be:
- explicit
- logged into config_resolved.json
- reflected in run_summary.json

---

## 6) Versioning

- scenario_version increments when schema changes
- output schema version increments when required fields change
- backwards compatibility policy:
  - v1 runner must be able to load scenario_version=1
  - breaking changes require new scenario_version
