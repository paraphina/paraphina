# Phase A-B2: Adversarial Search + Scenario Promotion

This document describes the adversarial / worst-case search system that discovers
failure scenarios and promotes them to a permanent regression suite.

## Overview

The adversarial search implements **evolutionary-lite / hill-climb over seeds**
(per WHITEPAPER Appendix B2) to find failure cases faster than random Monte Carlo.

Key goals:
- **Maximize failure likelihood**: Find scenarios that trigger kill events, large drawdowns, or worst-case PnL
- **Deterministic**: Fixed RNG seed produces identical results
- **Promote failures**: Discovered scenarios become permanent regression tests
- **Path-based scenarios**: Generated suite uses YAML file paths (no inline env_overrides)

## Quick Start

### Smoke Test (Fast)

```bash
# Run quick smoke test (3 candidates, fast ticks)
python3 -m batch_runs.phase_a.adversarial_search_promote --smoke --out runs/adv_smoke

# Run the generated suite
cargo run -p paraphina --bin sim_eval -- suite \
  scenarios/suites/adversarial_regression_v2.yaml \
  --output-dir runs/adv_reg_v2_smoke --verbose

# Verify evidence
cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_v2_smoke
```

### Full Search

```bash
# Run full adversarial search (20 trials)
python3 -m batch_runs.phase_a.adversarial_search_promote \
  --trials 20 \
  --ticks 300 \
  --top-k 10 \
  --seed 42 \
  --out runs/phaseA_adversarial_search

# Run regression suite
cargo run -p paraphina --bin sim_eval -- suite \
  scenarios/suites/adversarial_regression_v2.yaml \
  --output-dir runs/adv_reg_v2 --verbose

# Verify all evidence packs
cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_v2
```

## Output Directory Structure

```
<out_dir>/
├── search_trials.jsonl    # Every candidate (one JSON per line)
├── topk.json              # Selected top-K scenarios with metadata
├── summary.md             # Human-readable summary
└── runs/                  # Isolated output per candidate
    ├── cand_0001_s42/
    │   ├── scenario.yaml
    │   ├── evidence_pack/
    │   └── <run outputs>
    └── cand_0002_s43/
        └── ...
```

## Promoted Artifacts

When scenarios are promoted, they are written to:

```
scenarios/
├── v1/
│   └── adversarial/
│       └── generated_v1/
│           ├── adv_s00042_a1b2c3d4.yaml
│           ├── adv_s00043_e5f6g7h8.yaml
│           └── ...
└── suites/
    └── adversarial_regression_v2.yaml
```

### Suite v2 Format

The generated `adversarial_regression_v2.yaml` uses **path-based scenarios only**:

```yaml
suite_id: adversarial_regression_v2
suite_version: 2

scenarios:
  # Rank 1: score=350.00, seed=42
  - path: scenarios/v1/adversarial/generated_v1/adv_s00042_a1b2c3d4.yaml

  # Rank 2: score=300.00, seed=43
  - path: scenarios/v1/adversarial/generated_v1/adv_s00043_e5f6g7h8.yaml
```

This is fully compatible with `sim_eval suite --output-dir`.

## Adversarial Scoring

Candidates are scored to maximize failure likelihood:

```
score = 1000 * kill_switch + 10 * max_drawdown - 0.1 * mean_pnl
```

Priority order:
1. **Kill switch triggered** (highest weight: 1000)
2. **Large drawdown** (weight: 10)
3. **Negative PnL** (weight: -0.1, so lower PnL = higher score)

## Deterministic Guarantees

Given the same `--seed`:
- Same candidates are generated
- Same adversarial scores are computed
- Same top-K are selected
- Same scenario filenames are generated
- Same suite YAML is produced

Scenario filenames are deterministic:
```
adv_s{seed:05d}_{hash8}.yaml
```

The hash is derived from scenario parameters, ensuring stable filenames.

## Running with sim_eval

### Run the Suite

```bash
cargo run -p paraphina --bin sim_eval -- suite \
  scenarios/suites/adversarial_regression_v2.yaml \
  --output-dir runs/adv_reg_v2 \
  --verbose
```

### Verify Evidence Correctly

**IMPORTANT**: Use `verify-evidence-tree` on the **output root**, not on `evidence_pack/`:

```bash
# Correct: verify the output root
cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_v2

# Wrong: don't pass evidence_pack directory directly
# cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_v2/evidence_pack
```

### Run Individual Scenarios

```bash
cargo run -p paraphina --bin sim_eval -- run \
  scenarios/v1/adversarial/generated_v1/adv_s00042_a1b2c3d4.yaml \
  --output-dir runs/single_scenario \
  --verbose

cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/single_scenario
```

## Integration with Promotion Pipeline

The `promote_pipeline.py` automatically uses `adversarial_regression_v2.yaml` if available:

```python
# In evaluate_trial():
adversarial_suite, adv_version = get_adversarial_suite()
# Returns v2 if exists, else v1
```

This ensures:
- New v2 suite (path-based) is preferred
- Falls back to v1 (with env_overrides) if v2 not found
- Promotion pipeline never skips adversarial testing

## CLI Reference

```
python3 -m batch_runs.phase_a.adversarial_search_promote --help

Options:
  --smoke           Quick smoke mode (3 candidates, fast ticks)
  --trials N        Number of candidates to evaluate (default: 10)
  --ticks N         Ticks per scenario (default: 200)
  --top-k K         Number of top scenarios to promote (default: 5)
  --seed N          Base RNG seed (default: 42)
  --out DIR         Output directory
  --quiet, -q       Suppress verbose output
```

## Testing

Run hermetic unit tests (no cargo):

```bash
python3 -m unittest batch_runs.phase_a.tests.test_adversarial_search -v
```

Tests cover:
- Deterministic candidate generation
- Deterministic scenario naming (same seed = same filename)
- Top-K selection with tiebreaks
- Suite YAML generation (non-empty, sorted, path-based)
- Scenario promotion

## CI Integration

The adversarial regression is run in CI:

```yaml
# In .github/workflows/adversarial_regression.yml
- name: Run adversarial search (smoke)
  run: |
    python3 -m batch_runs.phase_a.adversarial_search_promote \
      --smoke --out runs/phaseA_adv_search_smoke_ci

- name: Run adversarial suite
  run: |
    cargo run -p paraphina --bin sim_eval -- suite \
      scenarios/suites/adversarial_regression_v2.yaml \
      --output-dir runs/adv_reg_v2_smoke_ci --verbose

- name: Verify evidence
  run: |
    cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_v2_smoke_ci
```

## Related Documentation

- [WHITEPAPER.md](WHITEPAPER.md) - Appendix B2: Stress + adversarial search
- [PHASE_A_PROMOTION_PIPELINE.md](PHASE_A_PROMOTION_PIPELINE.md) - Promotion pipeline
- [EVIDENCE_PACK.md](EVIDENCE_PACK.md) - Evidence pack format and verification
- [SIM_EVAL_SPEC.md](SIM_EVAL_SPEC.md) - sim_eval binary specification
- [ROADMAP.md](../ROADMAP.md) - Phase A status

