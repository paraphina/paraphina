# Phase B: Confidence-Aware Statistical Gating

## Overview

Phase B introduces statistically rigorous promotion criteria based on confidence intervals and dominance testing against a baseline. It builds on Phase A's multi-objective optimization by adding bootstrap-based confidence bounds to ensure promotion decisions are statistically sound.

## Tri-State Decision Model

Phase B uses a **tri-state decision model** with clear semantics:

| Decision | Meaning | When It Occurs |
|----------|---------|----------------|
| **PROMOTE** | Candidate is provably better | Guardrails pass AND promotion criteria pass |
| **HOLD** | Candidate is not worse, but not provably better | Guardrails pass BUT promotion criteria fail |
| **REJECT** | Candidate is provably worse | Guardrails fail |
| **ERROR** | Cannot evaluate | Missing data, parse errors, etc. |

### Guardrails vs Promotion Criteria

**Guardrails (Non-regression checks):**
- Pass when candidate is *not provably worse* than baseline
- CI overlap is acceptable (candidate may be equal or better)
- Failure → REJECT

**Promotion Criteria (Superiority checks):**
- Pass only when candidate is *provably better* than baseline at the configured alpha
- CI overlap causes these to fail (must demonstrate clear improvement)
- Failure (with guardrails passing) → HOLD

### Why HOLD is Common and Expected

**Smoke runs commonly yield HOLD, and that is expected and healthy.** With small sample sizes (e.g., 3-10 trials), confidence intervals are wide and often overlap even when the point estimates differ. This is the correct statistical behavior:

- HOLD means "we don't have enough evidence to promote, but also no evidence of regression"
- HOLD is a safe intermediate state that prevents premature promotion
- To achieve PROMOTE, collect more data to narrow confidence intervals

## Key Features

1. **Bootstrap-Based Confidence Intervals**: Uses block bootstrap to account for temporal dependencies in simulation data
2. **Two-Stage Gating**: Guardrails for safety, promotion criteria for superiority
3. **Conservative Design**: HOLD when uncertain, REJECT when provably worse
4. **Clear Decision Reasoning**: Human-readable explanations for all decisions
5. **Deterministic Execution**: Seeded RNG ensures reproducible results

## Architecture

```
batch_runs/phase_b/
├── __init__.py          # Public API
├── __main__.py          # Module entry point
├── cli.py               # Command-line interface
├── confidence.py        # Bootstrap & CI engine
├── compare.py           # Baseline comparison & dominance
├── gate.py              # Promotion gate logic
├── run_data.py          # Phase A trials discovery & JSONL loader
└── tests/
    ├── test_confidence.py  # Bootstrap CI stability tests
    ├── test_compare.py     # Dominance logic tests
    ├── test_gate.py        # Gate rejection reasoning tests
    └── test_run_data.py    # Trials discovery & parsing tests
```

## Modules

### confidence.py - Bootstrap & CI Engine

Implements:
- **Block Bootstrap**: Preserves temporal dependencies in time-series data
- **Estimators**: `compute_mean`, `compute_median`, `compute_cvar`, `compute_max_drawdown`
- **Percentile CIs**: Conservative confidence intervals via percentile method
- **Deterministic RNG**: Seeded `numpy.random.Generator` for reproducibility

```python
from batch_runs.phase_b.confidence import bootstrap_ci, compute_mean

data = pnl_values  # numpy array
lower, point, upper = bootstrap_ci(
    data, compute_mean,
    alpha=0.05,      # 95% CI
    n_bootstrap=1000,
    seed=42,
)
```

### compare.py - Baseline Comparison

Implements:
- **One-Sided Dominance**: Candidate must strictly beat baseline
- **Metric Directions**: Higher-is-better (PnL) vs lower-is-better (drawdown)
- **CI Overlap Tests**: Non-overlapping CIs indicate statistical significance
- **Threshold Checks**: Absolute thresholds (e.g., max kill rate)

**Dominance Rules:**
- PnL (higher is better): `candidate.lower_ci > baseline.upper_ci`
- Drawdown (lower is better): `candidate.upper_ci < baseline.lower_ci`
- Kill rate: `candidate.kill_rate <= threshold`

```python
from batch_runs.phase_b.compare import compare_runs

result = compare_runs(
    candidate_metrics,
    baseline_metrics,
    kill_threshold=0.10,
)
print(result.all_pass)      # True/False
print(result.fail_reasons)  # List of human-readable reasons
```

### run_data.py - Phase A Trials Loader

Implements:
- **Trials Discovery**: Find `trials.jsonl` with configurable precedence
- **JSONL Parsing**: Resilient parsing with blank line handling
- **Metric Extraction**: Try multiple key paths for each metric
- **Status Filtering**: Accept/reject based on status and is_valid fields
- **Detailed Errors**: Actionable error messages with file paths and rejection counts

```python
from batch_runs.phase_b.run_data import load_run_data

loaded = load_run_data(Path("runs/phaseA_candidate_smoke"))
print(f"Loaded {loaded.n_observations} observations from {loaded.trials_file}")
print(f"Rejections: {loaded.rejection_counts}")

# Convert to gate-compatible format
pnl_arrays = loaded.to_pnl_arrays()
kill_flags = loaded.to_kill_flags()
```

### gate.py - Promotion Gate

Implements:
- **Load Run Data**: Uses `run_data.py` for Phase A trials, with fallback to `mc_summary.json`
- **Metric Computation**: Bootstrap CIs for all key metrics
- **Tri-State Evaluation**: Separate guardrail and promotion criteria checks
- **Decision Object**: `PromotionDecision` with outcome, reasons, reports

```python
from batch_runs.phase_b.gate import evaluate_promotion

decision = evaluate_promotion(
    candidate_dir=Path("runs/phaseA_candidate_smoke"),
    baseline_dir=Path("runs/phaseA_baseline_smoke"),  # optional
    alpha=0.05,
    kill_threshold=0.10,
)

print(decision.outcome)           # PROMOTE / HOLD / REJECT / ERROR
print(decision.exit_code)         # 0=promote/hold, 2=reject, 3=error
print(decision.guardrails_passed) # True if candidate not provably worse
print(decision.promotion_passed)  # True if candidate provably better
print(decision.decision_reason)   # Human-readable explanation
```

## Input Format

### What `--candidate-run` and `--baseline-run` Must Point To

Both arguments accept:

1. **A Phase A run directory** containing `trials.jsonl` at the root or in a subdirectory
2. **A direct path to a `.jsonl` file** with trial records

Phase B reads `trials.jsonl` from each run directory. This file is produced by Phase A and contains one JSON object per line with trial metrics.

### Discovery Precedence

When given a directory path, Phase B discovers the trials file with this precedence:

1. If the path is a `.jsonl` file, use it directly
2. Else if `<path>/trials.jsonl` exists, use it
3. Else search recursively for `**/trials.jsonl` and pick the shallowest match (fewest path components), breaking ties by newest modification time

### Required Metrics in `trials.jsonl`

Each line must be a JSON object with these metrics (Phase B tries multiple key names for resilience):

| Metric | Primary Key | Fallback Keys |
|--------|-------------|---------------|
| PnL | `pnl` | `pnl_mean`, `mc_mean_pnl`, `mean_pnl`, `final_pnl` |
| Kill CI | `kill_ci` | `kill_ucb`, `mc_kill_prob_ci_upper`, `kill_prob_ci_upper`, `kill_rate` |
| Drawdown CVaR | `dd_cvar` | `mc_drawdown_cvar`, `drawdown_cvar` (optional) |

Metrics can also be nested under `metrics.<key>` or `result.<key>`.

### Status Filtering

Records are accepted if:
- No `status` field exists, OR
- `status` is one of: `OK`, `PASS`, `VALID`, `ok`, `pass`, `valid`, `true`

Records with `is_valid: false` are rejected.

## CLI Usage

```bash
# Compare candidate against baseline (Phase A run directories)
python3 -m batch_runs.phase_b.cli \
    --candidate-run runs/phaseA_candidate_smoke \
    --baseline-run runs/phaseA_baseline_smoke \
    --out-dir runs/phaseB_confidence_smoke \
    --alpha 0.05

# Evaluate with thresholds only (no baseline)
python3 -m batch_runs.phase_b.cli \
    --candidate-run runs/phaseA_candidate_smoke \
    --out-dir runs/gate_output \
    --kill-threshold 0.10 \
    --pnl-threshold 10.0

# Custom confidence level
python3 -m batch_runs.phase_b.cli \
    --candidate-run runs/candidate \
    --baseline-run runs/baseline \
    --out-dir runs/gate_output \
    --alpha 0.01 \
    --n-bootstrap 2000
```

### Smoke Run Example

To reproduce the standard smoke test pipeline:

```bash
# 1. Run Phase A to generate candidate and baseline
python3 -m batch_runs.phase_a.cli --study phaseA_candidate_smoke --profile balanced --n-trials 3
python3 -m batch_runs.phase_a.cli --study phaseA_baseline_smoke --profile balanced --n-trials 3

# 2. Run Phase B confidence gating
python3 -m batch_runs.phase_b.cli \
    --candidate-run runs/phaseA_candidate_smoke \
    --baseline-run runs/phaseA_baseline_smoke \
    --out-dir runs/phaseB_confidence_smoke
```

### Exit Codes (Institutional CI Semantics)
- `0`: PROMOTE or HOLD - Pipeline succeeded (HOLD means not enough evidence yet)
- `2`: REJECT - Candidate is worse / fails guardrails
- `3`: ERROR - Runtime/IO/parsing failure, data errors

### Outputs
- `confidence_report.json`: Machine-readable report with all metrics and decisions
- `confidence_report.md`: Human-readable Markdown report

## Integration with Phase A

Phase B is integrated into the Phase A promotion pipeline. Enable it with CLI flags:

```bash
python3 -m batch_runs.phase_a.promote_pipeline \
    --study my_study \
    --phase-b-enable \
    --phase-b-alpha 0.05 \
    --phase-b-n-bootstrap 1000 \
    --phase-b-kill-threshold 0.10
```

When Phase B is enabled, the pipeline:
1. Runs Phase A optimization and selects winners per tier
2. Before promoting each winner, runs Phase B gate
3. Only promotes if Phase B gate passes
4. Writes Phase B reports to `<trial>/phase_b_gate/`

## Decision Rules

### Guardrails (Must Pass to Avoid REJECT)

Guardrails check that candidate is **not provably worse** than baseline:

| Check | Rule | Passes When |
|-------|------|-------------|
| PnL non-regression | baseline does NOT dominate | `baseline.lower_ci <= candidate.upper_ci` |
| Drawdown non-regression | baseline does NOT dominate | `baseline.upper_ci >= candidate.lower_ci` |
| Kill rate threshold | rate within limit | `candidate.kill_rate <= threshold` |

If any guardrail fails → **REJECT** (candidate is provably worse).

### Promotion Criteria (Must Pass to Achieve PROMOTE)

Promotion criteria check that candidate is **provably better** than baseline:

| Check | Rule | Passes When |
|-------|------|-------------|
| PnL superiority | candidate dominates | `candidate.lower_ci > baseline.upper_ci` |
| Drawdown superiority | candidate dominates | `candidate.upper_ci < baseline.lower_ci` |

If all guardrails pass but any promotion criterion fails → **HOLD** (not provably better, needs more data).

If all guardrails AND all promotion criteria pass → **PROMOTE** (candidate is provably better).

### Kill Rate Threshold (if provided)
```
candidate.kill_rate <= threshold
```
Point estimate of kill rate must not exceed threshold. This is a guardrail check.

## Statistical Methods

### Block Bootstrap
Uses Künsch (1989) block bootstrap to account for temporal autocorrelation:
- Default block size: `sqrt(n)` observations
- Resamples contiguous blocks rather than individual observations
- Preserves serial correlation structure

### Confidence Intervals
Uses percentile method for CIs:
- Lower bound: `alpha/2` percentile of bootstrap distribution
- Upper bound: `1 - alpha/2` percentile
- Conservative for asymmetric distributions

### Estimators
- **Mean**: Arithmetic mean of final PnL per run
- **Median**: Median of final PnL per run (robust)
- **CVaR**: Expected value of worst `alpha` fraction (tail risk)
- **Max Drawdown**: Largest peak-to-trough decline

## Tests

Run Phase B tests:

```bash
# All Phase B tests
python3 -m pytest batch_runs/phase_b/tests/ -v

# Specific test files
python3 -m pytest batch_runs/phase_b/tests/test_confidence.py -v  # Bootstrap stability
python3 -m pytest batch_runs/phase_b/tests/test_compare.py -v     # Dominance logic
python3 -m pytest batch_runs/phase_b/tests/test_gate.py -v        # Gate reasoning
```

### Key Test Cases
- `test_bootstrap_ci_stability`: Same seed → same CI
- `test_dominance_logic`: Correct dominance detection
- `test_gate_rejection_reasoning`: Clear human-readable reasons
- `test_identical_metrics_yields_hold`: Same candidate/baseline → HOLD
- `test_candidate_provably_worse_yields_reject`: Worse candidate → REJECT
- `test_candidate_dominates_all_metrics_yields_promote`: Better candidate → PROMOTE

## Configuration

### PromotionGate Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.05 | Significance level for CIs |
| `n_bootstrap` | int | 1000 | Number of bootstrap samples |
| `block_size` | int | None | Block size (None = sqrt(n)) |
| `seed` | int | 42 | Random seed for reproducibility |
| `kill_threshold` | float | None | Max allowed kill rate |
| `drawdown_threshold` | float | None | Max allowed drawdown |
| `pnl_threshold` | float | None | Min required PnL lower CI |
| `require_strict_dominance` | bool | True | Require candidate to strictly dominate |

## Dependencies

- `numpy`: Array operations and random number generation
- No scipy required (pure numpy implementation)
- Python 3.8+ (uses `statistics.NormalDist`)

## Design Principles

1. **Deterministic**: Same seed produces same results
2. **Conservative**: HOLD when uncertain, REJECT when provably worse, PROMOTE only when provably better
3. **Transparent**: Clear guardrail and promotion criteria in reports
4. **Two-Stage Gating**: Separate non-regression (guardrails) from superiority (promotion)
5. **Reproducible**: All parameters logged in reports

## JSON Report Schema

The `confidence_report.json` includes these key fields:

```json
{
  "decision": "HOLD",           // PROMOTE | HOLD | REJECT | ERROR
  "outcome": "hold",            // lowercase version
  "guardrails_passed": true,    // Did non-regression checks pass?
  "promotion_passed": false,    // Did superiority checks pass?
  "decision_reason": "HOLD: Guardrails passed but promotion criteria failed...",
  "guardrail_checks": [
    {"name": "PnL non-regression", "passed": true, "reason": "..."},
    {"name": "Drawdown non-regression", "passed": true, "reason": "..."}
  ],
  "promotion_checks": [
    {"name": "PnL superiority", "passed": false, "reason": "..."},
    {"name": "Drawdown superiority", "passed": false, "reason": "..."}
  ],
  "exit_code": 0,               // 0=promote/hold, 2=reject, 3=error
  "alpha": 0.05,
  "n_bootstrap": 1000,
  "seed": 42,
  "candidate_samples": 3,
  "baseline_samples": 3
}
```

