# Phase A Promotion Pipeline

This document describes the Phase A-2 multi-objective tuning and promotion pipeline.

## Overview

The promotion pipeline implements a deterministic, budget-gated research→promotion loop:

1. **Generate Candidates** - Seeded random sampling + optional mutation around Pareto set
2. **Evaluate Each Candidate** - Monte Carlo + suite gating + evidence verification
3. **Compute Pareto Frontier** - Multi-objective optimization over pnl/risk metrics
4. **Promote Winners** - Budget-filtered selection per risk tier

All operations are deterministic given a fixed seed.

## Quick Start

### Smoke Run (Fast Test)

```bash
# Run smoke test with 3 trials, 10 MC runs, 100 ticks
python3 -m batch_runs.phase_a.promote_pipeline --smoke --study-dir runs/phaseA_smoke
```

This produces:
- `runs/phaseA_smoke/trials.jsonl` - One JSON per trial
- `runs/phaseA_smoke/pareto.json` - Pareto frontier candidates
- `runs/phaseA_smoke/pareto.csv` - Pareto frontier as CSV
- `configs/presets/promoted/<tier>/phaseA_*_<timestamp>.env` - Promoted presets
- `configs/presets/promoted/<tier>/PROMOTION_RECORD.json` - Promotion provenance

### Full Research Run

```bash
# 50 trials with default MC settings
python3 -m batch_runs.phase_a.promote_pipeline --trials 50 --study research_v1

# Customize MC parameters
python3 -m batch_runs.phase_a.promote_pipeline \
  --trials 100 \
  --mc-runs 200 \
  --mc-ticks 1200 \
  --seed 42 \
  --study full_research
```

## Command Line Options

```
python3 -m batch_runs.phase_a.promote_pipeline --help

Options:
  --smoke             Quick smoke mode (3 trials, 10 MC runs, 100 ticks)
  --study NAME        Study name (default: auto-generated timestamp)
  --study-dir DIR     Override study directory path
  --trials N          Number of candidate trials (default: 10)
  --mc-runs N         Monte Carlo runs per candidate (default: 50)
  --mc-ticks N        Ticks per Monte Carlo run (default: 600)
  --seed N            Random seed for determinism (default: 42)
  --budgets PATH      Path to budgets.yaml (default: batch_runs/phase_a/budgets.yaml)
  --quiet, -q         Suppress verbose output
```

## Artifacts

### Trial Directory Structure

Each trial creates an isolated directory:

```
runs/phaseA/<study>/<trial_id>/
├── candidate.env           # Configuration overrides
├── mc/                     # Monte Carlo outputs
│   ├── mc_summary.json     # Aggregated metrics + tail risk
│   ├── monte_carlo.yaml    # Run configuration
│   └── evidence_pack/      # Evidence Pack v1
│       ├── SHA256SUMS
│       └── manifest.json
└── oos/                    # Out-of-sample suite results
    ├── research_v1/
    └── adversarial_regression_v1/
```

### Study-Level Outputs

```
runs/phaseA/<study>/
├── trials.jsonl            # All trial results (one JSON per line)
├── pareto.json             # Pareto frontier with full metadata
└── pareto.csv              # Pareto frontier as CSV
```

### Promoted Presets

```
configs/presets/promoted/<tier>/
├── phaseA_<study>_<timestamp>.env    # Config for sourcing
└── PROMOTION_RECORD.json             # Full provenance record
```

## How Promotion Works

### 1. Candidate Evaluation Contract

For each candidate:

1. Create isolated trial directory
2. Write `candidate.env` with configuration overrides
3. Run `monte_carlo` binary with env overlay → `mc/`
4. Verify evidence pack: `sim_eval verify-evidence-tree <trial>/mc/`
5. Run out-of-sample suite: `sim_eval suite scenarios/suites/research_v1.yaml`
6. Run adversarial suite: `sim_eval suite scenarios/suites/adversarial_regression_v1.yaml`
7. Parse metrics from `mc_summary.json`

**Trial fails if evidence verification fails.**

### 2. Multi-Objective Optimization

Objectives (Pareto optimization):
- **Maximize** `mean_pnl`
- **Minimize** `kill_prob_ci_upper` (95% Wilson CI upper bound)
- **Minimize** `drawdown_cvar` (CVaR at 95%)

A candidate is on the Pareto frontier if no other valid candidate dominates it on all objectives.

### 3. Budget Gating

Budgets are defined in `batch_runs/phase_a/budgets.yaml`:

```yaml
conservative:
  max_kill_prob: 0.05
  max_drawdown_cvar: 500.0
  min_mean_pnl: 10.0

balanced:
  max_kill_prob: 0.10
  max_drawdown_cvar: 1000.0
  min_mean_pnl: 20.0

aggressive:
  max_kill_prob: 0.15
  max_drawdown_cvar: 2000.0
  min_mean_pnl: 30.0
```

A candidate passes a tier budget if:
- `kill_prob_ci_upper <= max_kill_prob`
- `drawdown_cvar <= max_drawdown_cvar`
- `mean_pnl >= min_mean_pnl`
- Evidence verification passed
- Research suite passed
- Adversarial suite passed

### 4. Winner Selection

For each tier, select the single winner using deterministic tie-breaking:

1. Filter candidates by budget constraints
2. Sort by `mean_pnl` descending (highest first)
3. Tie-break: lowest `drawdown_cvar`
4. Tie-break: lowest `kill_prob_ci_upper`
5. Final tie-break: `candidate_id` alphabetically

### 5. Promotion Output

Promoted configs are written to `configs/presets/promoted/<tier>/`:

**Config file** (`phaseA_<study>_<timestamp>.env`):
```bash
# Promoted configuration for balanced tier
# Study: research_v1
# Candidate: abc123def456
# Metrics: pnl=45.2340, kill_ci=0.0623

export PARAPHINA_RISK_PROFILE=balanced
export PARAPHINA_HEDGE_BAND_BASE=0.05
export PARAPHINA_MM_SIZE_ETA=1.0
...
```

**Promotion record** (`PROMOTION_RECORD.json`):
```json
{
  "schema_version": 1,
  "promoted_at": "2025-12-24T20:30:00.000Z",
  "study_name": "research_v1",
  "tier": "balanced",
  "git_commit": "abc123...",
  "candidate_id": "abc123def456",
  "candidate_hash": "<sha256 of config file>",
  "trial_id": "trial_0005_abc123def456",
  "trial_dir": "runs/phaseA/research_v1/trial_0005_abc123def456",
  "config": { ... },
  "env_overlay": { ... },
  "commands_run": [
    "monte_carlo --runs 50 --seed 47",
    "sim_eval suite scenarios/suites/research_v1.yaml",
    "sim_eval suite scenarios/suites/adversarial_regression_v1.yaml"
  ],
  "seeds": [47],
  "evidence_verified": true,
  "metrics": {
    "mean_pnl": 45.234,
    "pnl_cvar": -123.45,
    "drawdown_cvar": 456.78,
    "kill_prob_point": 0.04,
    "kill_prob_ci_upper": 0.0623,
    "total_runs": 50
  },
  "suites_passed": {
    "research": true,
    "adversarial": true
  }
}
```

## Reproducing a Promoted Candidate

To reproduce a promoted candidate's run:

```bash
# 1. Source the promoted config
source configs/presets/promoted/balanced/phaseA_research_v1_20251224_203000.env

# 2. Re-run monte_carlo with the same seed
cargo run --release -p paraphina --bin monte_carlo -- \
  --runs 50 \
  --ticks 600 \
  --seed 47 \
  --output-dir runs/reproduce_balanced

# 3. Verify the evidence pack matches
cargo run --release -p paraphina --bin sim_eval -- \
  verify-evidence-tree runs/reproduce_balanced

# 4. Compare mc_summary.json metrics
diff runs/phaseA/research_v1/trial_0005_abc123def456/mc/mc_summary.json \
     runs/reproduce_balanced/mc_summary.json
```

The results should be identical given:
- Same git commit
- Same seed
- Same configuration

## Testing

Run unit tests (hermetic, no cargo):

```bash
# All phase_a tests
python3 -m pytest batch_runs/phase_a/tests/ -v

# Specific test files
python3 -m pytest batch_runs/phase_a/tests/test_pareto.py -v
python3 -m pytest batch_runs/phase_a/tests/test_budgets.py -v
python3 -m pytest batch_runs/phase_a/tests/test_winner_selection.py -v
python3 -m pytest batch_runs/phase_a/tests/test_env_parsing.py -v
```

Or with unittest:

```bash
python3 -m unittest discover -s batch_runs/phase_a/tests -v
```

## Dependencies

- Python 3.7+ (stdlib only, no pandas)
- Optional: PyYAML (for parsing suite files; built-in parser works without it)
- Rust binaries: `monte_carlo`, `sim_eval` (built via `cargo build --release`)

## Architecture

```
batch_runs/phase_a/
├── __init__.py
├── __main__.py          # Entry point
├── promote_pipeline.py  # Main CLI and pipeline logic
├── cli.py              # Legacy CLI (optimize/promote subcommands)
├── evaluate.py         # Candidate evaluation
├── optimize.py         # Pareto + winner selection
├── promote.py          # Promotion logic
├── schemas.py          # Data types
├── budgets.yaml        # Default budget definitions
└── tests/
    ├── test_pareto.py
    ├── test_budgets.py
    ├── test_winner_selection.py
    └── test_env_parsing.py
```

## Related Documentation

- [WHITEPAPER.md](WHITEPAPER.md) - Strategy specification
- [EVIDENCE_PACK.md](EVIDENCE_PACK.md) - Evidence pack format
- [SIM_EVAL_SPEC.md](SIM_EVAL_SPEC.md) - sim_eval binary specification
- [ROADMAP.md](../ROADMAP.md) - Development roadmap (Phase A checklist)

