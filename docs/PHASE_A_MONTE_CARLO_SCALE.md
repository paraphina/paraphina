# Phase A: Monte Carlo at Scale

This document describes the deterministic sharding and aggregation system for running Monte Carlo simulations at scale.

## Overview

The MC Scale system enables running large Monte Carlo studies (thousands to millions of runs) by:
1. **Deterministic sharding**: Splitting runs into independent shards that can be executed in parallel
2. **Aggregation**: Combining shard results into a single canonical output
3. **Evidence verification**: Ensuring integrity through cryptographic hashes and evidence packs

## Deterministic Seed Contract

**Non-negotiable invariant**: For global run index `i`, the scenario seed is:

```
seed_i = base_seed + i (u64 wrapping add)
```

This ensures:
- Same `(seed, runs)` always produces identical results
- Shards can be computed independently and aggregated deterministically
- Results are reproducible across machines and time

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         mc_scale.py                             │
│  plan → run-shard → aggregate → evidence verification           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     monte_carlo binary                          │
│  --run-start-index, --run-count, --seed → mc_runs.jsonl         │
│  summarize --input → mc_summary.json                            │
└─────────────────────────────────────────────────────────────────┘
```

## Plan Schema

The `mc_scale_plan.json` file defines the sharding configuration:

```json
{
  "schema_version": 1,
  "seed": 12345,
  "runs": 1000,
  "shards": 10,
  "ticks": 600,
  "out_dir": "/path/to/output",
  "shard_ranges": [
    {"start": 0, "end": 100},
    {"start": 100, "end": 200},
    ...
  ]
}
```

Fields:
- `schema_version`: Plan schema version (currently 1)
- `seed`: Base seed for deterministic seed mapping
- `runs`: Total number of runs across all shards
- `shards`: Number of shards (may be less if runs < requested shards)
- `ticks`: Ticks per run
- `out_dir`: Absolute path to output directory
- `shard_ranges`: List of `{start, end}` pairs (end is exclusive)

**Determinism guarantee**: Same arguments produce byte-identical plan files (stable key ordering, stable formatting).

## JSONL Output Format

Each shard produces `mc_runs.jsonl` with one JSON record per line:

```json
{"run_index": 0, "seed": 12345, "pnl_total": 100.5, "max_drawdown": 10.2, "kill_switch": false, "kill_tick": null, "kill_reason": "None", "ticks_executed": 600, "max_abs_delta_usd": 50.0, "max_abs_basis_usd": 25.0, "max_abs_q_tao": 1.5, "max_venue_toxicity": 0.1}
```

Required fields:
- `run_index`: Global run index (0-based)
- `seed`: Seed used for this run (must equal `base_seed + run_index`)
- `pnl_total`: Final PnL
- `max_drawdown`: Maximum drawdown during run
- `kill_switch`: Whether kill switch was triggered

## Evidence Pack Structure

After aggregation, the output directory contains:

```
<out_dir>/
├── mc_scale_plan.json          # Original plan
├── mc_runs.jsonl               # Aggregated JSONL (all runs)
├── mc_summary.json             # Summary statistics (from Rust summarize)
├── mc_scale_manifest.json      # Hash manifest for verification
├── evidence_pack/
│   ├── manifest.json           # Evidence pack manifest
│   ├── suite.yaml              # For verifier compatibility
│   └── SHA256SUMS              # Checksums
└── shards/
    ├── shard_0/
    │   ├── mc_runs.jsonl
    │   ├── mc_summary.json
    │   ├── monte_carlo.yaml
    │   └── evidence_pack/
    ├── shard_1/
    │   └── ...
    └── ...
```

## Usage

### Local Development

```bash
# Build the binaries
cargo build --release -p paraphina --bin monte_carlo --bin sim_eval

# Run smoke test (small defaults)
python3 -m batch_runs.phase_a.mc_scale smoke \
  --seed 12345 \
  --runs 12 \
  --shards 3 \
  --ticks 50 \
  --out-dir runs/local/mc_scale_smoke

# Or run steps individually:

# 1. Generate plan
python3 -m batch_runs.phase_a.mc_scale plan \
  --out-dir runs/mc_scale \
  --seed 42 \
  --runs 1000 \
  --shards 10 \
  --ticks 600

# 2. Run each shard (can be parallelized)
for i in $(seq 0 9); do
  python3 -m batch_runs.phase_a.mc_scale run-shard \
    --plan runs/mc_scale/mc_scale_plan.json \
    --shard-index $i
done

# 3. Aggregate results
python3 -m batch_runs.phase_a.mc_scale aggregate \
  --plan runs/mc_scale/mc_scale_plan.json
```

### CI Usage

The `mc_scale_smoke.yml` workflow runs on every PR and push to main:

```yaml
- name: Run MC scale smoke test
  run: |
    python3 -m batch_runs.phase_a.mc_scale smoke \
      --seed 12345 \
      --runs 12 \
      --shards 3 \
      --ticks 50 \
      --out-dir runs/ci/mc_scale_smoke
```

### Parallel Execution (Advanced)

For large-scale runs, execute shards in parallel:

```bash
# Generate plan
python3 -m batch_runs.phase_a.mc_scale plan \
  --out-dir runs/mc_scale_large \
  --seed 42 \
  --runs 10000 \
  --shards 100 \
  --ticks 1200

# Run shards in parallel (e.g., using GNU parallel)
seq 0 99 | parallel -j 8 \
  python3 -m batch_runs.phase_a.mc_scale run-shard \
    --plan runs/mc_scale_large/mc_scale_plan.json \
    --shard-index {}

# Aggregate
python3 -m batch_runs.phase_a.mc_scale aggregate \
  --plan runs/mc_scale_large/mc_scale_plan.json
```

## Rust Binary Interface

### Run Mode (Sharded)

```bash
monte_carlo \
  --runs 1000 \                    # Total runs (for reference)
  --run-start-index 0 \            # Start at global index 0
  --run-count 100 \                # Execute 100 runs
  --seed 42 \                      # Base seed
  --ticks 600 \                    # Ticks per run
  --output-dir runs/shard_0 \      # Output directory
  --quiet                          # Suppress per-run output
```

Outputs:
- `mc_runs.jsonl`: Per-run JSONL records
- `mc_summary.json`: Summary statistics for this shard
- `monte_carlo.yaml`: Configuration used
- `evidence_pack/`: Evidence pack for verification

### Summarize Mode

```bash
monte_carlo summarize \
  --input runs/aggregated/mc_runs.jsonl \
  --out-dir runs/aggregated \
  --base-seed 42                   # Optional: validate seed contract
```

Validation performed:
1. No duplicate `run_index` values
2. Contiguous indices (no gaps)
3. If `--base-seed` provided: validates `seed == base_seed + run_index`

## Validation Rules

The aggregator enforces these rules:

1. **Exactly `runs` records**: The aggregated JSONL must have exactly the number of records specified in the plan
2. **Contiguous indices**: Run indices must be `[0, runs)` with no gaps
3. **No duplicates**: Each `run_index` appears exactly once
4. **Seed contract**: If base seed is known, each `seed == base_seed + run_index`

## Integration with Phase AB Promotion

The MC Scale output (`mc_summary.json`) is compatible with the Phase AB promotion pipeline. The tail risk metrics computed by `monte_carlo summarize` are the single source of truth for:

- `kill_probability.ci_upper` (Wilson 95% CI upper bound)
- `max_drawdown_var_cvar.cvar` (CVaR at 95%)
- `pnl.mean` (mean PnL)

These metrics are used by the Phase AB gate to determine PROMOTE/HOLD/REJECT decisions.

## Troubleshooting

### Binary not found

```
Error: monte_carlo binary not found
```

Build the binary:
```bash
cargo build --release -p paraphina --bin monte_carlo
```

### Shard mismatch

```
Error: Expected 1000 records, found 998
```

Check that all shards completed successfully. Re-run failed shards:
```bash
python3 -m batch_runs.phase_a.mc_scale run-shard \
  --plan runs/mc_scale/mc_scale_plan.json \
  --shard-index <failed_shard>
```

### Seed contract violation

```
Error: Seed mismatch at run_index 500
```

Verify that the same `--seed` was used for all shards and that no shard was run with incorrect parameters.

