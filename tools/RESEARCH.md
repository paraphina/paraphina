# Paraphina Research & Calibration Harness

This document explains how to reproduce the research experiments and calibration runs we’ve built
around the Paraphina engine.

All commands are run from the project root:

```bash
cd ~/code/paraphina
source .venv/bin/activate

## World-model alignment pipeline (Exp10–Exp12)

This section documents the risk-profile alignment pipeline that connects:

- the world-model budgets (risk tiers in `exp07_optimal_presets.RISK_TIERS`),
- the calibration / validation experiments (Exp06–Exp09),
- the research JSONL datasets (manual / ad-hoc runs),
- and the live Rust presets in `Config::for_profile` (`src/config.rs`).

The goal is to make it trivial to answer:

> “Do the live **conservative / balanced / aggressive** profiles still behave
> according to their world-model risk budgets, given the latest research runs?”

### 10. Exp10 – Calibration + validation vs world model

**Script:** `tools/exp10_world_model_alignment.py`  
**Input:**

- `runs/exp07_optimal_presets/exp07_presets.csv`
- `runs/exp08_profile_validation/exp08_profile_summary.csv`

**Output:**

- `runs/exp10_world_model_alignment/exp10_alignment.csv`
- Console summary per risk tier.

This compares the **design-time** metrics (from the Exp07 grid and Exp08 validation)
against the world-model budgets from `RISK_TIERS`:

- `max_kill_prob`
- `max_drawdown_abs`
- `min_final_pnl_mean`

For each tier it reports:

- whether design metrics satisfy the budgets (`design_*_ok`),
- whether validation metrics satisfy the budgets (`val_*_ok`),
- margins vs budget (positive = safe side),
- and a risk-adjusted alignment score.

**How to run:**

```bash
cd ~/code/paraphina
python tools/exp10_world_model_alignment.py
