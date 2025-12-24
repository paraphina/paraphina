# Paraphina Research & Calibration Harness

This document explains how to reproduce the research experiments and calibration
runs we’ve built around the Paraphina engine.

All commands assume you are in the project root and the Python venv is active:

    cd ~/code/paraphina
    source .venv/bin/activate

Rust binaries are built with:

    cargo build --release

---

## 0. Core telemetry & research tools

These helpers are used by multiple experiments and by the higher-level
world-model pipeline.

### 0.1 Tick-level telemetry from the Rust engine

The Rust binary can emit per-tick telemetry as JSONL. Set:

- `PARAPHINA_TELEMETRY_MODE = jsonl`
- `PARAPHINA_TELEMETRY_PATH = <path to .jsonl>`

Example (balanced profile, 2000 ticks, manual test):

    export PARAPHINA_TELEMETRY_MODE=jsonl
    export PARAPHINA_TELEMETRY_PATH=runs/manual_telemetry/test_run.jsonl

    cargo run --release -- --ticks 2000 --profile balanced

This will create `runs/manual_telemetry/test_run.jsonl`.

### 0.2 `tools/research_ticks.py`

Condense a single tick-level telemetry JSONL file into **one row** of a research
JSONL dataset.

Typical usage:

    python3 tools/research_ticks.py \
      --telemetry runs/manual_telemetry/test_run.jsonl \
      --profile balanced \
      --label manual_balanced_001 \
      --out research_dataset_v001.jsonl

The script:

- loads the ticks via `batch_runs.ts_metrics.load_telemetry_jsonl`,
- computes PnL / drawdown / regime fractions / kill-switch,
- appends one JSON object to the output dataset,
- prints a human-readable summary.

### 0.3 `tools/research_summary.py`

Summarise a research JSONL dataset produced by `research_ticks.py`.

Example:

    python3 tools/research_summary.py \
      --input research_dataset_v001.jsonl

Outputs:

- a **global** summary (runs, mean PnL, mean max drawdown, kill fraction),
- a per-profile summary,
- and the top runs ranked by a simple utility score.

### 0.4 `tools/manual_telemetry_summary.py`

Helper for quickly summarising a single telemetry run without adding it to a
dataset. Useful when manually testing a new config or profile.

---

## 1. Calibration & preset design (Exp02–Exp09)

These experiments search over configuration space and validate candidate
profiles. They are typically run occasionally (not every day).

Very high-level overview:

- **Exp02 – Hedge-band sweep / heatmap**  
  Script: `tools/exp02_hedge_band_heatmap.py`  
  Sweeps hedge-band parameters over a grid and plots PnL vs band.

- **Exp03 – Size / η / vol sweep**  
  Script: `tools/exp03_size_eta_vol_sweep.py`  
  Explores the interaction between trade size, η (aggressiveness) and vol
  scaling.

- **Exp04 – Risk-regime sweep**  
  Script: `tools/exp04_risk_regime_sweep.py`  
  Stress-tests how often the engine visits Normal / Warning / HardLimit regimes
  under different thresholds.

- **Exp05 – Calibration report**  
  Script: `tools/exp05_calibration_report.py`  
  Produces a consolidated report for a baseline config.

- **Exp06 – Full research suite**  
  Script: `tools/exp06_full_research_suite.py`  
  Orchestrates the earlier experiments over a baseline config.

- **Exp07 – Optimal presets & world-model risk tiers**  
  Script: `tools/exp07_optimal_presets.py`  
  Defines `RISK_TIERS` (world-model budgets) and chooses candidate presets for
  each tier.  
  Output: `runs/exp07_optimal_presets/exp07_presets.csv`.

- **Exp08 – Profile validation**  
  Script: `tools/exp08_profile_validation.py`  
  Validates candidate presets via simulation.  
  Output: `runs/exp08_profile_validation/exp08_profile_summary.csv`.

- **Exp09 – Hyperparameter search**  
  Script: `tools/exp09_hyperparam_search.py`  
  Optional extra search over strategy hyperparameters.

Most of these scripts can be run simply as:

    python3 tools/exp0X_*.py

They all write their outputs under `runs/` with an `exp0X_` prefix. See the
docstring in each script for detailed options.

These experiments feed into the **world-model alignment pipeline** below.

---

## 2. World-model alignment pipeline (Exp10–Exp13)

This pipeline connects:

- the world-model budgets (risk tiers in `exp07_optimal_presets.RISK_TIERS`),
- the calibration / validation experiments (Exp06–Exp09),
- the research JSONL datasets (manual or automated runs),
- and the live Rust presets in `Config::for_profile` (`src/config.rs`).

The goal is to make it trivial to answer:

> Do the live **conservative / balanced / aggressive** profiles still behave
> according to their world-model risk budgets, given the latest research runs?

We break this into four steps: Exp10, Exp11, Exp12, and Exp13.

---

## Exp10 – Calibration + validation vs world model

**Script:** `tools/exp10_world_model_alignment.py`  

**Inputs:**

- `runs/exp07_optimal_presets/exp07_presets.csv`
- `runs/exp08_profile_validation/exp08_profile_summary.csv`

**Outputs:**

- `runs/exp10_world_model_alignment/exp10_alignment.csv`
- Console summary per risk tier.

What it does:

- For each risk tier in `RISK_TIERS` it compares:

  - **Design metrics** (from Exp07 presets),
  - **Validation metrics** (from Exp08 simulations),

  against the world-model budgets:

  - `max_kill_prob`
  - `max_drawdown_abs`
  - `min_final_pnl_mean`

- Reports:

  - whether design metrics satisfy the budgets (`design_*_ok`),
  - whether validation metrics satisfy the budgets (`val_*_ok`),
  - margins vs budget (positive = safe side),
  - and a risk-adjusted alignment score.

How to run:

    cd ~/code/paraphina
    python3 tools/exp10_world_model_alignment.py

---

## Exp11 – Research vs world-model alignment

**Script:** `tools/exp11_research_alignment.py`  

**Inputs:**

- A research JSONL dataset produced by `research_ticks.py`, e.g.
  `research_dataset_v003.jsonl` or `research_dataset_v004.jsonl`.

**Outputs:**

- `runs/exp11_research_alignment/exp11_alignment.csv`
- Console report per profile (and optionally per label).

What it does:

- Aggregates PnL / drawdown / kill-switch stats from the dataset.
- Groups by `profile` (and label if present).
- Compares empirical metrics to the `RISK_TIERS` budgets.
- Reports booleans (`pnl_ok`, `dd_ok`, `kill_ok`) and margins, e.g.:

  - Are we within max drawdown?
  - Is kill probability below budget?
  - Is mean PnL above the required minimum?

Simple run over the whole dataset:

    python3 tools/exp11_research_alignment.py \
      --dataset research_dataset_v004.jsonl

You can also restrict to a subset of runs with:

- `--profile` to choose a single profile.
- `--label-contains SUBSTR` to filter labels.

---

## Exp12 – Profile safety tuner

**Script:** `tools/exp12_profile_saftey_tuner.py`  
(*Note: the filename contains a typo, kept for compatibility.*)

**Inputs:**

- `runs/exp11_research_alignment/exp11_alignment.csv`

**Outputs:**

- `runs/exp12_profile_saftey_tuner/exp12_tuned_presets.csv`
- `runs/exp12_profile_saftey_tuner/exp12_tuning_report.csv`
- Console summary of proposed changes.

What it does:

- Looks at the Exp11 alignment results per profile.
- For each profile, estimates how “tight” or “loose” the empirical risk is
  versus the world-model budgets.
- Proposes **safer live presets** for:

  - `daily_loss_limit`
  - max drawdown budgets
  - and related risk parameters,

  such that we maintain target behaviour with a margin of safety.

How to run:

    python3 tools/exp12_profile_saftey_tuner.py

The resulting `exp12_tuned_presets.csv` is a reference when updating
`Config::for_profile` in `src/config.rs`.

---

## Exp13 – Auto research harness (profiles)

**Script:** `tools/exp13_auto_research_harness.py`  

**Purpose:** Run the full research loop across risk profiles in a single
command, and check world-model alignment on the resulting dataset.

### Exp13 pipeline

For each risk profile `p` in `{conservative, balanced, aggressive}`:

1. **Run the Rust engine with telemetry enabled**

   Environment:

   - `PARAPHINA_TELEMETRY_MODE = jsonl`
   - `PARAPHINA_TELEMETRY_PATH = runs/exp13_auto/<profile>_r<idx>.jsonl`

   Command (launched by the script):

       cargo run --release -- --ticks <T> --profile <profile>

2. **Condense telemetry into a research row**

   For each telemetry file, the script calls:

       python3 tools/research_ticks.py \
         --telemetry runs/exp13_auto/<profile>_r<idx>.jsonl \
         --profile <profile> \
         --label <label_prefix>_<profile>_rXXX \
         --out <dataset>

   This appends one JSON object per run into the chosen research dataset,
   e.g. `research_dataset_v004.jsonl`.

After all runs finish:

3. **Summarise the research dataset**

       python3 tools/research_summary.py \
         --input <dataset>

   This prints the global and per-profile summaries and ranks the top runs by
   utility.

4. **Run world-model alignment on the dataset**

       python3 tools/exp11_research_alignment.py \
         --dataset <dataset>

   This checks whether the empirical behaviour in the dataset is consistent
   with the world-model budgets for each profile.

The script itself orchestrates all of these steps and prints a concise
end-to-end report.

### Exp13 usage

Minimal example (1 run per profile, 1000 ticks):

    python3 tools/exp13_auto_research_harness.py \
      --ticks 1000 \
      --runs-per-profile 1 \
      --dataset research_dataset_v004.jsonl

Heavier example (3 runs per profile, 2000 ticks):

    python3 tools/exp13_auto_research_harness.py \
      --ticks 2000 \
      --runs-per-profile 3 \
      --dataset research_dataset_v004.jsonl

### Exp13 arguments

- `--ticks`  
  Number of ticks per run (default: `2000`).

- `--runs-per-profile`  
  Number of independent repeats per risk profile (default: `2`).

- `--dataset`  
  Research JSONL dataset to append to (default:
  `research_dataset_v004.jsonl`).

- `--profiles`  
  Which profiles to run (default: `conservative balanced aggressive`).

- `--label-prefix`  
  Prefix for labels stored in the dataset (default: `exp13`).

---

## 3. Typical daily workflow

A typical “is everything still safe?” check might look like:

1. Pull latest code and ensure Rust compiles:

       cd ~/code/paraphina
       git pull
       cargo build --release

2. Run Exp13 over the current profiles:

       python3 tools/exp13_auto_research_harness.py \
         --ticks 2000 \
         --runs-per-profile 2 \
         --dataset research_dataset_v004.jsonl

3. Inspect:

   - `research_summary.py` output for PnL / drawdown quality.
   - `exp11_research_alignment.py` output for world-model alignment.
   - `exp12_profile_saftey_tuner` results if you want to adjust live
     presets.

This gives a reproducible loop from world-model budgets through
simulated research back to the live Rust configuration.
