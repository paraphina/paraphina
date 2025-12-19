# Simulation Output Schema (v1)

This document defines the *required* output fields for simulation runs.

## run_summary.json (required)
Top-level:
- scenario_id: string
- scenario_version: int
- seed: int
- build_info:
  - git_sha: string
  - dirty: bool
- config:
  - risk_profile: string
  - init_q_tao: number
  - dt_seconds: number
  - steps: int
- results:
  - final_pnl_usd: number
  - max_drawdown_usd: number
  - kill_switch:
      - triggered: bool
      - step: int|null
      - reason: string|null
- determinism:
  - checksum: string

## metrics.jsonl (optional)
Each line is a JSON object:
- step: int
- pnl_usd: number
- inventory_q_tao: number
- sigma_eff: number|null
- spread_quoted: number|null
- fills: { maker: int, taker: int } (if available)
