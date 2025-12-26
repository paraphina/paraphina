# Phase A-B2: CEM Adversarial Search + Scenario Promotion

This document describes the Cross-Entropy Method (CEM) adversarial search system
that discovers failure scenarios and optionally promotes them to a permanent regression suite.

## Overview

The adversarial search implements **institutional-grade CEM search** over adversarial
scenario parameters (per WHITEPAPER Appendix B2):

- **CEM iterations**: Maintains mean/std per parameter, samples population, selects elite
- **Bounded sampling**: All parameters clipped to allowed ranges
- **Deterministic**: Fixed RNG seed produces identical search trajectory
- **Output isolation**: All artifacts under `runs/` by default (git-clean workflow)
- **Explicit promotion**: Only writes to `scenarios/` when `--promote-suite` is passed

### Key Design Principles

1. **Institutional hygiene**: Generated artifacts never dirty git working tree unless explicit
2. **Non-empty guarantee**: Suite generation always produces at least 1 scenario (with fallbacks)
3. **Deterministic tie-breaking**: Same seed = same results = same scenario filenames
4. **Self-contained output**: Generated suite works from the output directory

## Quick Start

### Smoke Test (Fast, Output Isolated)

```bash
# Run quick smoke test - all outputs under runs/
rm -rf runs/phaseA_adv_search_smoke
python3 -m batch_runs.phase_a.adversarial_search_promote \
  --smoke --out runs/phaseA_adv_search_smoke

# Run the generated suite
cargo run -p paraphina --bin sim_eval -- suite \
  runs/phaseA_adv_search_smoke/generated_suite/adversarial_regression_generated.yaml \
  --output-dir runs/adv_reg_cem_smoke --verbose

# Verify evidence
cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_cem_smoke
```

### CEM Search (Full)

```bash
# Run CEM search (5 iterations, 20 candidates each)
python3 -m batch_runs.phase_a.adversarial_search_promote \
  --iterations 5 \
  --pop-size 20 \
  --elite-frac 0.2 \
  --ticks 300 \
  --top-k 10 \
  --seed 42 \
  --out runs/phaseA_adv_search_cem
```

### Explicit Promotion (Modifies Repo)

```bash
# Only when ready to commit promoted scenarios:
python3 -m batch_runs.phase_a.adversarial_search_promote \
  --iterations 3 \
  --pop-size 15 \
  --top-k 10 \
  --promote-suite

# This writes to:
#   scenarios/v1/adversarial/promoted_v1/*.yaml
#   scenarios/suites/adversarial_regression_v3.yaml
#   scenarios/suites/PROMOTION_RECORD.json
```

## Output Directory Structure

### Default (under `runs/`)

```
runs/phaseA_adv_search/<study_id>/
├── candidates/                    # Scenario YAML files for top-K
│   ├── adv_s00042_a1b2c3d4.yaml
│   ├── adv_s00043_e5f6g7h8.yaml
│   └── ...
├── generated_suite/               # Self-contained suite
│   └── adversarial_regression_generated.yaml
├── runs/                          # Per-candidate run outputs
│   ├── cem_i00_c0000_s42/
│   │   ├── scenario.yaml
│   │   └── <sim_eval outputs>
│   └── ...
├── search_results.jsonl           # All candidates (one JSON per line)
└── summary.json                   # CEM stats + top-K
```

### Promoted (with `--promote-suite`)

```
scenarios/
├── v1/
│   └── adversarial/
│       └── promoted_v1/
│           ├── adv_s00042_a1b2c3d4.yaml
│           └── ...
└── suites/
    ├── adversarial_regression_v3.yaml
    └── PROMOTION_RECORD.json
```

## CEM Algorithm

The Cross-Entropy Method maintains a distribution over scenario parameters:

1. **Initialize**: Mean = center of bounds, Std = 1/4 of range
2. **Sample**: Draw N candidates from N(mean, std), clip to bounds
3. **Evaluate**: Run sim_eval for each candidate
4. **Score**: `score = 1000*kill + 10*max_drawdown - 0.1*mean_pnl`
5. **Select elite**: Top 20% by adversarial score
6. **Update**: Shift mean/std toward elite (learning rate = 0.5)
7. **Repeat**: For I iterations

### Parameter Space

| Parameter | Bounds | Description |
|-----------|--------|-------------|
| vol | [0.005, 0.05] | Base volatility |
| vol_multiplier | [0.5, 3.0] | Volatility scaling |
| jump_intensity | [0.0001, 0.002] | Jump events per step |
| jump_sigma | [0.01, 0.10] | Jump magnitude |
| spread_bps | [0.5, 5.0] | Spread in basis points |
| latency_ms | [0.0, 50.0] | Latency |
| init_q_tao | [-60.0, 60.0] | Initial inventory |
| daily_loss_limit | [300.0, 3000.0] | Loss limit |

## Adversarial Scoring

Candidates are scored to maximize failure likelihood:

```
score = 1000 * kill_switch + 10 * max_drawdown - 0.1 * mean_pnl
```

Priority order:
1. **Kill switch triggered** (weight: 1000)
2. **Large drawdown** (weight: 10)
3. **Negative PnL** (weight: -0.1)

## Deterministic Guarantees

Given the same `--seed`:
- Same CEM trajectory (same candidates per iteration)
- Same adversarial scores computed
- Same top-K selected
- Same scenario filenames generated
- Same suite YAML produced

Scenario filenames are deterministic:
```
adv_s{seed:05d}_{hash8}.yaml
```

## Non-Empty Suite Guarantee

The suite generation always produces at least 1 scenario:

1. If CEM produces valid elite → use top-K
2. If elite empty but valid results → use top-1 from all results
3. If all results fail → use deterministic safe fallback scenario

## Running with sim_eval

### Run Generated Suite

```bash
cargo run -p paraphina --bin sim_eval -- suite \
  runs/phaseA_adv_search_smoke/generated_suite/adversarial_regression_generated.yaml \
  --output-dir runs/adv_reg_test \
  --verbose
```

### Verify Evidence Correctly

**IMPORTANT**: Use `verify-evidence-tree` on the **output root**, not on `evidence_pack/`:

```bash
# Correct: verify the output root
cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_test

# Wrong: don't pass evidence_pack directory directly
# cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_test/evidence_pack
```

## CLI Reference

```
python3 -m batch_runs.phase_a.adversarial_search_promote --help

Options:
  --smoke             Quick smoke mode (3 fixed candidates, no CEM)
  --iterations N      Number of CEM iterations (default: 3)
  --pop-size N        Population size per iteration (default: 20)
  --elite-frac F      Fraction of population to use as elite (default: 0.2)
  --ticks N           Ticks per scenario (default: 200)
  --top-k K           Number of top scenarios in final suite (default: 5)
  --seed N            Base RNG seed (default: 42)
  --out DIR           Output directory (default: runs/phaseA_adv_search/<timestamp>)
  --promote-suite     Promote top-K to scenarios/ and create v3 suite (MODIFIES REPO)
  --quiet, -q         Suppress verbose output
```

## Testing

Run hermetic unit tests (no cargo):

```bash
python3 -m unittest discover -s batch_runs/phase_a/tests -p "test_*.py" -v
```

Or specific test module:

```bash
python3 -m unittest batch_runs.phase_a.tests.test_adversarial_search -v
```

Tests cover:
- CEM distribution initialization and update determinism
- CEM candidate generation determinism
- Deterministic scenario naming
- Adversarial scoring priority
- Elite selection
- Non-empty suite generation guarantees
- Suite YAML format validation

## CI Integration

The adversarial regression is run in CI:

```yaml
# In .github/workflows/adversarial_regression.yml
- name: Run CEM adversarial search (smoke)
  run: |
    python3 -m batch_runs.phase_a.adversarial_search_promote \
      --smoke --out runs/phaseA_adv_search_smoke_ci

- name: Run generated suite
  run: |
    cargo run -p paraphina --bin sim_eval -- suite \
      runs/phaseA_adv_search_smoke_ci/generated_suite/adversarial_regression_generated.yaml \
      --output-dir runs/adv_reg_cem_smoke_ci --verbose

- name: Verify evidence
  run: |
    cargo run -p paraphina --bin sim_eval -- verify-evidence-tree runs/adv_reg_cem_smoke_ci
```

## Promotion Workflow

When you're ready to promote discovered scenarios to the repo:

1. Run full CEM search with good parameters
2. Review `summary.json` and top-K scenarios
3. Run the generated suite and verify evidence
4. If satisfied, run with `--promote-suite`:

```bash
# Generate with promotion
python3 -m batch_runs.phase_a.adversarial_search_promote \
  --iterations 5 --pop-size 30 --top-k 10 --promote-suite

# Verify promotion
ls scenarios/v1/adversarial/promoted_v1/
cat scenarios/suites/PROMOTION_RECORD.json

# Commit
git add scenarios/
git commit -m "Promote CEM adversarial scenarios v3"
```

The `PROMOTION_RECORD.json` contains:
- Git SHA at promotion time
- Command line used
- Base seed
- Top-K count
- Scoring info
- Source run directory

## Related Documentation

- [WHITEPAPER.md](WHITEPAPER.md) - Appendix B2: Stress + adversarial search
- [PHASE_A_PROMOTION_PIPELINE.md](PHASE_A_PROMOTION_PIPELINE.md) - Promotion pipeline
- [EVIDENCE_PACK.md](EVIDENCE_PACK.md) - Evidence pack format and verification
- [SIM_EVAL_SPEC.md](SIM_EVAL_SPEC.md) - sim_eval binary specification
- [ROADMAP.md](../ROADMAP.md) - Phase A status
