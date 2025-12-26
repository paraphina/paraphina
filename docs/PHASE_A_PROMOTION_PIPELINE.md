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

ADR (Adversarial Delta Regression) Options:
  --adr-enable                      Enable ADR gating
  --adr-suite PATH                  Suite path for ADR (repeatable)
  --adr-baseline-cache DIR          Baseline cache directory
  --adr-gate-max-regression-usd N   Maximum allowed regression in USD
  --adr-gate-max-regression-pct N   Maximum allowed regression as percentage
  --adr-write-md                    Write ADR report in Markdown (default: on)
  --adr-no-write-md                 Disable Markdown report output
  --adr-write-json                  Write ADR report in JSON (default: on)
  --adr-no-write-json               Disable JSON report output
```

## ADR (Adversarial Delta Regression) Gating

ADR provides institutional-grade regression gating by comparing candidate trial outputs
against cached baseline outputs using `sim_eval report`.

### What ADR Does

1. **Baseline Caching**: For each (profile, suite) combination, a baseline suite run is
   cached under `<study_dir>/_baseline_cache/<profile>/<suite_name>/`. This baseline is
   reused across all trials with the same profile.

2. **Report Generation**: After each candidate's suite run, `sim_eval report` compares
   the candidate outputs against the cached baseline, generating:
   - `<trial_dir>/report/<suite_name>.md` - Markdown report
   - `<trial_dir>/report/<suite_name>.json` - JSON report data

3. **Regression Gating**: If `--adr-gate-max-regression-usd` or `--adr-gate-max-regression-pct`
   is specified, the report command enforces these as hard gates. Candidates that exceed
   the regression threshold fail ADR gating and are excluded from promotion.

### ADR Quick Start

```bash
# Run smoke test with ADR enabled
python3 -m batch_runs.phase_a.promote_pipeline --smoke --adr-enable \
  --study-dir runs/phaseA_smoke_adr

# Run with ADR and regression gates
python3 -m batch_runs.phase_a.promote_pipeline --smoke --adr-enable \
  --adr-gate-max-regression-usd 50.0 \
  --adr-gate-max-regression-pct 10.0 \
  --study-dir runs/phaseA_adr_gated
```

### How Baseline Caching Works

The baseline cache path is computed deterministically:

```
<study_dir>/_baseline_cache/<profile>/<suite_stem>/
```

For example:
- Study dir: `runs/phaseA_smoke_adr`
- Profile: `balanced`
- Suite: `scenarios/suites/research_v1.yaml`
- Baseline cache: `runs/phaseA_smoke_adr/_baseline_cache/balanced/research_v1/`

The baseline is created once per (profile, suite) combination and reused for all
subsequent trials. This ensures:
- Deterministic comparisons (same baseline for all candidates)
- Efficient runtime (baseline only generated once)
- Reproducibility (baseline artifacts are preserved)

### ADR Output Structure

When ADR is enabled, additional outputs are generated:

```
runs/phaseA/<study>/
├── _baseline_cache/                    # Cached baselines (reused across trials)
│   ├── balanced/
│   │   ├── research_v1/               # Baseline for balanced + research_v1
│   │   │   ├── evidence_pack/
│   │   │   └── <scenario_outputs>/
│   │   └── adversarial_regression_v2/ # Baseline for balanced + adversarial
│   ├── conservative/
│   │   └── ...
│   └── aggressive/
│       └── ...
├── trial_0000_<hash>/
│   ├── suite/                         # Candidate suite outputs
│   │   ├── research_v1/
│   │   └── adversarial_regression_v2/
│   └── report/                        # ADR reports
│       ├── research_v1.md
│       ├── research_v1.json
│       ├── adversarial_regression_v2.md
│       └── adversarial_regression_v2.json
└── ...
```

### Reproducing an ADR Report

To reproduce an ADR report from artifacts:

```bash
# 1. Identify baseline and candidate directories from trials.jsonl
# 2. Run sim_eval report manually
cargo run --release -p paraphina --bin sim_eval -- report \
  --baseline runs/phaseA_smoke_adr/_baseline_cache/balanced/research_v1 \
  --variant candidate=runs/phaseA_smoke_adr/trial_0000_abc123/suite/research_v1 \
  --out-md report_reproduced.md \
  --out-json report_reproduced.json

# 3. Compare with original report
diff runs/phaseA_smoke_adr/trial_0000_abc123/report/research_v1.json report_reproduced.json
```

### ADR and Smoke Mode

In smoke mode, ADR still exercises the full path but with minimal scenarios/seeds.
This ensures the ADR integration is tested without exploding runtime.

## Artifacts

### Trial Directory Structure

Each trial creates an isolated directory with institutional-grade output isolation:

```
runs/phaseA/<study>/<trial_id>/
├── candidate.env                           # Configuration overrides
├── mc/                                     # Monte Carlo outputs
│   ├── mc_summary.json                     # Aggregated metrics + tail risk
│   ├── monte_carlo.yaml                    # Run configuration
│   └── evidence_pack/                      # Evidence Pack v1
│       ├── SHA256SUMS
│       └── manifest.json
└── suite/                                  # Suite outputs (using --output-dir)
    ├── research_v1/                        # Research suite output
    │   ├── evidence_pack/                  # Suite evidence pack
    │   │   ├── SHA256SUMS
    │   │   └── manifest.json
    │   └── <scenario_outputs>/
    └── adversarial_regression_v1/          # Adversarial suite output
        ├── evidence_pack/                  # Suite evidence pack
        │   ├── SHA256SUMS
        │   └── manifest.json
        └── <scenario_outputs>/
```

**IMPORTANT**: Suites are ALWAYS run with `--output-dir` to ensure:
- All artifacts are isolated under the trial directory
- Evidence packs are verifiable with a single command
- No cross-contamination between trial runs

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

1. Create isolated trial directory: `runs/phaseA/<study>/<trial_id>/`
2. Write `candidate.env` with configuration overrides
3. Run `monte_carlo` binary with env overlay → `<trial>/mc/`
4. Verify MC evidence pack: `sim_eval verify-evidence-tree <trial>/mc/`
5. Run research suite WITH --output-dir:
   ```bash
   sim_eval suite scenarios/suites/research_v1.yaml \
     --output-dir <trial>/suite/research_v1 --verbose
   ```
6. Verify research evidence: `sim_eval verify-evidence-tree <trial>/suite/research_v1`
7. Run adversarial suite WITH --output-dir (NEVER SKIPPED):
   ```bash
   sim_eval suite scenarios/suites/adversarial_regression_v1.yaml \
     --output-dir <trial>/suite/adversarial_regression_v1 --verbose
   ```
8. Verify adversarial evidence: `sim_eval verify-evidence-tree <trial>/suite/adversarial_regression_v1`
9. Final verification of entire trial: `sim_eval verify-evidence-tree <trial>/`
10. Parse metrics from `mc_summary.json`

**Trial fails if ANY evidence verification fails.**
**Adversarial suite is NEVER skipped - missing suite file = hard failure.**

### Suite env_overrides Merging

When running suites with inline `env_overrides` (e.g., adversarial scenarios):

1. **Candidate env** (from `candidate.env`) provides base configuration
2. **Suite scenario env_overrides** can override any candidate setting
3. This allows adversarial scenarios to stress-test specific conditions

Priority (highest to lowest):
1. Suite scenario `env_overrides` (stress test params)
2. Candidate config env overlay (tuning params)
3. Process environment

### Verification Commands

After a trial completes, verify evidence packs:

```bash
# Verify entire trial directory (all evidence packs)
sim_eval verify-evidence-tree runs/phaseA/<study>/<trial_id>/

# Verify just the MC evidence pack
sim_eval verify-evidence-pack runs/phaseA/<study>/<trial_id>/mc/

# Verify just a suite evidence pack
sim_eval verify-evidence-pack runs/phaseA/<study>/<trial_id>/suite/research_v1/

# Verify entire study directory
sim_eval verify-evidence-tree runs/phaseA/<study>/
```

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
- ADR gating passed (if `--adr-enable` is set)

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
  "schema_version": 2,
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
    "monte_carlo --runs 50 --seed 47 --output-dir <trial>/mc/",
    "sim_eval suite research_v1.yaml --output-dir <trial>/suite/research_v1 --verbose",
    "sim_eval suite adversarial_regression_v1.yaml --output-dir <trial>/suite/adversarial_regression_v1 --verbose",
    "sim_eval verify-evidence-tree <trial>/",
    "sim_eval report --baseline ... --variant candidate=... --out-md ... --out-json ..."
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
    "adversarial": true,
    "adr": true
  },
  "adr_results": [
    {
      "suite_name": "research_v1",
      "gates_passed": true,
      "report_md": "<trial>/report/research_v1.md",
      "report_json": "<trial>/report/research_v1.json"
    }
  ]
}
```

Note: `schema_version: 2` indicates the record includes ADR fields. The `adr_results` field
is only present when ADR gating was enabled.

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
python3 -m pytest batch_runs/phase_a/tests/test_adr.py -v
```

Or with unittest:

```bash
python3 -m unittest discover -s batch_runs/phase_a/tests -v
```

### ADR Tests

The ADR tests verify:
- Baseline cache path computation is stable and deterministic
- Report command construction is correct
- Evidence root selection works on synthetic directory trees

```bash
python3 -m unittest batch_runs.phase_a.tests.test_adr -v
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
├── promote_pipeline.py  # Main CLI and pipeline logic (includes ADR)
├── cli.py              # Legacy CLI (optimize/promote subcommands)
├── evaluate.py         # Candidate evaluation
├── optimize.py         # Pareto + winner selection
├── promote.py          # Promotion logic
├── schemas.py          # Data types
├── budgets.yaml        # Default budget definitions
└── tests/
    ├── test_adr.py           # ADR gating tests
    ├── test_pareto.py
    ├── test_budgets.py
    ├── test_winner_selection.py
    └── test_env_parsing.py
```

## Verification

The promotion pipeline automatically generates a **root evidence pack** after completing all trials. This provides cryptographic integrity for all pipeline outputs.

### Output Structure with Evidence Pack

```
runs/phaseA_smoke/
├── evidence_pack/                 # Root evidence pack (auto-generated)
│   ├── SHA256SUMS                 # Checksums for all study files
│   ├── manifest.json              # Structured metadata
│   └── suite.yaml                 # For verifier compatibility
├── trial_0000_*/                  # Trial directories
│   ├── mc/                        # Monte Carlo outputs
│   │   └── evidence_pack/         # Per-MC evidence pack
│   ├── suite/                     # Suite outputs
│   │   └── */evidence_pack/       # Per-suite evidence packs
│   └── report/                    # ADR reports (if enabled)
├── _baseline_cache/               # ADR baseline cache (if enabled)
├── trials.jsonl                   # All trial results
├── pareto.json                    # Pareto frontier
└── pareto.csv                     # Pareto as CSV
```

### Verification Commands

```bash
# Verify the study output root (root evidence pack)
sim_eval verify-evidence-pack runs/phaseA_smoke

# Verify all evidence packs (root + all nested trial/suite packs)
sim_eval verify-evidence-tree runs/phaseA_smoke
```

In smoke mode, the pipeline automatically verifies the root evidence pack after generation.

### Reproducing from Artifacts

Given a verified study directory:

1. The `trials.jsonl` contains complete trial results with all metrics
2. The `pareto.json` contains the Pareto frontier candidates
3. Each trial's `mc/` and `suite/` directories contain verifiable outputs
4. The `evidence_pack/SHA256SUMS` provides integrity verification

```bash
# 1. Verify the evidence pack
sim_eval verify-evidence-pack runs/phaseA_study/

# 2. Verify all nested packs
sim_eval verify-evidence-tree runs/phaseA_study/

# 3. Reproduce the Monte Carlo for a specific trial
sim_eval run <trial_dir>/scenario.yaml --output-dir runs/reproduce_mc

# 4. Compare results (should be deterministic)
diff runs/original/trial_0000*/mc/summary.json runs/reproduce_mc/summary.json
```

## Related Documentation

- [WHITEPAPER.md](WHITEPAPER.md) - Strategy specification
- [EVIDENCE_PACK.md](EVIDENCE_PACK.md) - Evidence pack format
- [SIM_EVAL_SPEC.md](SIM_EVAL_SPEC.md) - sim_eval binary specification
- [ROADMAP.md](../ROADMAP.md) - Development roadmap (Phase A checklist)

