# Experiments in Paraphina

This document standardises how experiments are run, what artefacts they produce,
and which output/metric contracts must remain stable.

Paraphina currently supports two complementary experiment styles:

1) **Stdout-metrics batch runs** (fast, minimal instrumentation)
   - implemented via `batch_runs/orchestrator.py` + `batch_runs/metrics.py`
   - canonical example: `batch_runs/exp03_stress_search.py`

2) **Telemetry→research dataset runs** (tick-level telemetry condensed to a dataset)
   - implemented via `tools/exp13_auto_research_harness.py`
   - condensed via `tools/research_ticks.py`
   - summarised/aligned via `tools/research_summary.py` + `tools/exp11_research_alignment.py`
   - tuned presets via `tools/exp12_profile_safety_tuner.py`

The rule: **every experiment must produce tidy artefacts under `runs/` and/or
append to a versioned dataset file**. No “only printed to console” results.

---

## 1. Directory conventions

### 1.1 Source code locations

- `batch_runs/`
  - experiment scripts that run the engine multiple times using stdout parsing
  - shared runner: `batch_runs/orchestrator.py`
  - shared metrics: `batch_runs/metrics.py`

- `tools/`
  - analysis and research pipelines (telemetry processing, preset selection, alignment)
  - examples:
    - `tools/exp07_optimal_presets.py`
    - `tools/exp11_research_alignment.py`
    - `tools/exp12_profile_safety_tuner.py`
    - `tools/exp13_auto_research_harness.py`

### 1.2 Output locations

All experiment artefacts go under:

- `runs/<exp_name>/...`

Recommended naming:

- Per-run results: `runs/<exp_name>/<exp_name>_runs.csv`
- Grouped summary: `runs/<exp_name>/<exp_name>_summary.csv`
- Research alignment: `runs/exp11_research_alignment/exp11_alignment.csv`
- Tuned presets: `runs/exp12_profile_safety_tuner/exp12_tuned_presets.csv`
- Presets selection: `runs/exp07_optimal_presets/exp07_presets.csv` + `.json`
- Telemetry JSONL: `runs/exp13_auto/<profile>_r<idx>.jsonl`

If you must write elsewhere, add a doc note and a symlink under `runs/`.

---

## 2. Experiment contracts (do not break casually)

### 2.1 Stdout summary contract (parsed by `batch_runs/metrics.py`)

The shared parser expects end-of-run stdout to contain a human-readable summary
including:

- Daily PnL line:
  - `Daily PnL (realised / unrealised / total): +123.45 / -67.89 / +55.56`

- Kill switch line (either form is acceptable):
  - `Kill switch: true`
  - `Kill switch active: false`

If you change the summary string format, you **must**:
- update `batch_runs/metrics.py` regexes, and
- add/adjust tests (recommended) to prevent silent parsing failures.

### 2.2 Per-run DataFrame contract (orchestrator)

`batch_runs/orchestrator.py` flattens:
- `label` (user-provided metadata dict), plus
- `metrics` (dict returned by `parse_metrics(stdout)`)

into a tidy per-run DataFrame via `results_to_dataframe(...)`.

Rules:
- labels must be JSON-serialisable (str/int/float/bool)
- avoid nested dicts in label keys
- keep metric key names stable across experiments if possible

### 2.3 Telemetry contract (research pipeline)

The telemetry-based harness enables tick-level JSONL telemetry via env vars:

- `PARAPHINA_TELEMETRY_MODE=jsonl`
- `PARAPHINA_TELEMETRY_PATH=<absolute path to .jsonl>`

`tools/research_ticks.py` then:
- reads the JSONL,
- condenses it into a *single* “research row” appended to a dataset JSONL.

Downstream consumers (notably `tools/exp11_research_alignment.py`) expect that
each dataset record contains (at minimum) fields equivalent to:

- profile identifier:
  - `profile` (preferred), or candidates like `risk_profile`, `tier`

- final PnL:
  - `final_pnl` (preferred), or candidates like `pnl_total`

- drawdown:
  - `max_drawdown` (preferred)

- kill indicator / probability:
  - `kill_switch` or `p_kill_switch` or similar (script supports multiple candidates)

If you rename dataset fields, ensure all tools are updated or preserve backward
compatibility by emitting both fields for a transition period.

---

## 3. Label conventions (recommended)

To keep runs comparable and joins easy, include at least:

### 3.1 For stdout-metrics batch_runs experiments
- `experiment`: e.g. `"exp03_stress_search"`
- `profile`: `"conservative" | "balanced" | "aggressive"`
- `repeat`: integer seed/repeat index
- any swept knobs as explicit label keys:
  - `init_q_tao`
  - `band_base`
  - `mm_size_eta`
  - `vol_ref`
  - `daily_loss_limit`
  - etc.

### 3.2 For telemetry-based research runs
`tools/research_ticks.py` already records:
- `profile`
- `label` (run label string)
- condensed metrics

Recommended label format:
- `<prefix>_<profile>_r<idx>`
  - example: `exp13_balanced_r001`

---

## 4. Environment-variable knobs (sweepable)

The orchestrator explicitly documents common env knobs. Keep these stable and
use them in experiments where possible:

- note: exact semantics depend on the Rust config/env parser.

Common knobs:
- `PARAPHINA_RISK_PROFILE`
- `PARAPHINA_INIT_Q_TAO`
- `PARAPHINA_HEDGE_BAND_BASE`
- `PARAPHINA_HEDGE_MAX_STEP`
- `PARAPHINA_MM_SIZE_ETA`
- `PARAPHINA_VOL_REF`
- `PARAPHINA_DAILY_LOSS_LIMIT`

Telemetry knobs:
- `PARAPHINA_TELEMETRY_MODE`
- `PARAPHINA_TELEMETRY_PATH`

Guideline:
- **prefer env knobs for experiments** over hardcoding values in Rust, so batch
  sweeps remain simple and reproducible.

---

## 5. How to run experiments

### 5.1 Stdout-based experiment (batch_runs)
Example pattern (see `batch_runs/exp03_stress_search.py`):

1) Ensure the binary exists:
   - typically `target/release/paraphina`

2) Run the experiment:
   - `python batch_runs/exp03_stress_search.py`

Outputs (expected):
- `runs/exp03_stress_search/exp03_stress_runs.csv`
- `runs/exp03_stress_search/exp03_stress_summary.csv`

### 5.2 Telemetry-based research harness (tools)
Example pattern (see `tools/exp13_auto_research_harness.py`):

- `python tools/exp13_auto_research_harness.py --ticks 2000 --runs-per-profile 2 --dataset research_dataset_vXXX.jsonl`

Outputs (expected):
- telemetry:
  - `runs/exp13_auto/<profile>_r<idx>.jsonl`
- dataset:
  - appended `research_dataset_vXXX.jsonl`
- summary + alignment:
  - console summary from `tools/research_summary.py`
  - `runs/exp11_research_alignment/exp11_alignment.csv`

---

## 6. Standard metrics to report

### 6.1 Stdout-metrics (fast loop)
From `batch_runs/metrics.py`:
- `pnl_realised`
- `pnl_unrealised`
- `pnl_total`
- `kill_switch` (boolean)

Recommended aggregations per (experiment, profile, swept_knobs...):
- mean/std of `pnl_total`
- kill fraction: mean of `kill_switch`
- run count

### 6.2 Research dataset metrics (telemetry condensed)
Minimum recommended:
- `final_pnl`
- `max_drawdown` (store as signed or magnitude, but be consistent)
- `kill_switch` or `p_kill_switch`
- (optional but encouraged)
  - average/global inventory stats
  - volatility stats
  - number of hedge actions / exit actions

---

## 7. “Golden” artefacts per experiment (recommended)

Every experiment should produce at least:

1) **Per-run CSV**
   - one row per simulation
   - includes labels + parsed metrics

2) **Grouped summary CSV**
   - grouped by the key swept parameters
   - includes:
     - pnl mean/std
     - kill fraction
     - run count

Optional but high-value:
- a simple notebook or script producing plots:
  - `pnl_total_mean` vs swept knob
  - `kill_switch_frac` vs swept knob
  - “efficient frontier” style scatter: pnl vs kill prob

---

## 8. Versioning and reproducibility

### 8.1 Datasets
Research datasets should be versioned by filename:
- `research_dataset_v001.jsonl`, `v002`, ...

Do not edit past versions in place.

### 8.2 Config traceability
Every run should be attributable to:
- git commit hash (ideal: embed into stdout/telemetry)
- risk profile
- key knobs (env overlays)
- run label

### 8.3 When you must change schemas
If you must rename fields:
- keep old fields for one release cycle OR provide a migration script
- update all consumers (`tools/*`, `batch_runs/*`) in the same PR

---

## 9. Adding a new experiment checklist

- [ ] Script lives in `batch_runs/` (stdout-based) or `tools/` (telemetry/research-based)
- [ ] Uses `EngineRunConfig` + `run_many` if it’s a multi-run sweep
- [ ] Writes under `runs/<exp_name>/`
- [ ] Produces `*_runs.csv` and `*_summary.csv` (or a dataset JSONL + clear summary outputs)
- [ ] Uses stable label keys (`experiment`, `profile`, `repeat`, swept knobs)
- [ ] Does not break stdout parsing (`batch_runs/metrics.py`) without updating it
- [ ] Documents any new env knobs and output columns

---

## 10. Related docs
- Target algorithm + implementation-truth framing:
  - `docs/WHITEPAPER.md`
- Engineering process for AI-assisted changes:
  - `docs/AI_PLAYBOOK.md`
- Plan of record:
  - `ROADMAP.md`

---
