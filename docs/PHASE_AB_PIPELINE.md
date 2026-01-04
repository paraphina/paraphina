# Phase AB Pipeline

**Phase AB** is an institutional-grade orchestrator that eliminates user error when running Phase B confidence gating on Phase A run outputs.

## Overview

Phase AB integrates:
1. **Run Root Validation**: Validates and normalizes Phase A run directories
2. **Evidence Verification**: Verifies evidence packs for data integrity
3. **Phase B Gating**: Runs confidence-aware statistical gating
4. **Canonical Outputs**: Writes a manifest and stable outputs

## Quick Start

### Run Mode (Explicit Paths)

```bash
python3 -m batch_runs.phase_ab.cli run \
    --candidate-run runs/phaseA_candidate \
    --baseline-run runs/phaseA_baseline \
    --out-dir runs/phaseAB_output
```

### Smoke Mode (CI-Friendly)

```bash
# Uses standard smoke directories
python3 -m batch_runs.phase_ab.cli smoke

# Auto-generate Phase A runs if they don't exist
python3 -m batch_runs.phase_ab.cli smoke --auto-generate-phasea
```

## What Phase AB Does

### 1. Run Root Resolution

Phase AB validates input directories and finds the canonical run root containing `trials.jsonl`:

| Input | Resolution |
|-------|------------|
| `<dir>/trials.jsonl` exists | Use `<dir>` |
| `<dir>/*/trials.jsonl` (one found) | Use nested directory |
| `<dir>/*/trials.jsonl` (multiple) | Use newest, print warning |
| No `trials.jsonl` found | Error with diagnosis |

**Diagnostic Error Messages:**

If the input looks like Phase A adversarial search output (contains `generated_suite/`, `search_results.jsonl`):

```
ERROR: This looks like Phase A adversarial search output, not an evaluated run.
Phase B gating requires an evaluated run containing trials.jsonl.

To create an evaluated run, run sim_eval suite using the generated suite yaml:
    sim_eval suite <dir>/generated_suite/suite.yaml --output-dir <output_path>
```

### 2. Evidence Verification

If the run root contains `evidence_pack/SHA256SUMS`, Phase AB verifies data integrity:

```bash
sim_eval verify-evidence-pack <run_root>
```

Use `--skip-evidence-verify` to disable (not recommended for production).

### 3. Phase B Gating

Phase AB calls the existing Phase B confidence gate logic:

- Bootstrap-based confidence intervals
- Tri-state decision model (PROMOTE/HOLD/REJECT)
- Statistical dominance testing

### 4. Canonical Outputs

Phase AB writes:

| File | Description |
|------|-------------|
| `phase_ab_manifest.json` | Canonical manifest with all metadata |
| `confidence_report.json` | Machine-readable Phase B report |
| `confidence_report.md` | Human-readable Phase B report |

## Manifest Schema

```json
{
  "schema_version": 1,
  "candidate_run_resolved": "/path/to/candidate",
  "baseline_run_resolved": "/path/to/baseline",
  "phase_b_out_dir": "/path/to/output",
  "decision": "HOLD",
  "alpha": 0.05,
  "bootstrap_samples": 1000,
  "seed": 42,
  "confidence_report_json": "/path/to/confidence_report.json",
  "confidence_report_md": "/path/to/confidence_report.md",
  "timestamp": "2025-01-04T12:00:00Z",
  "git_commit": "abc123...",
  "candidate_samples": 10,
  "baseline_samples": 10,
  "evidence_verified_candidate": true,
  "evidence_verified_baseline": true,
  "errors": []
}
```

## Exit Codes (CI Semantics)

Phase AB uses institutional CI exit code semantics:

| Exit Code | Decision | Meaning |
|-----------|----------|---------|
| 0 | PROMOTE | Candidate is provably better |
| 0 | HOLD | Pipeline succeeded; not enough evidence yet |
| 2 | REJECT | Candidate fails guardrails |
| 3 | ERROR | Runtime/IO/parsing failure |

**Key insight:** HOLD is exit code 0 because the pipeline *succeeded*—there's just not enough evidence to promote. This is expected for smoke runs with few trials.

## What HOLD Means

A **HOLD** decision indicates:

1. ✅ All guardrails passed (candidate not provably worse)
2. ⚠️ Promotion criteria failed (confidence intervals overlap)
3. 📊 More data needed to prove superiority

**HOLD is normal for smoke runs.** With 3-10 trials, confidence intervals are wide and rarely achieve the statistical significance needed for PROMOTE.

To achieve PROMOTE:
1. Collect more trials (narrow confidence intervals)
2. Or, demonstrate larger effect size

## CLI Reference

### `run` Command

```bash
python3 -m batch_runs.phase_ab.cli run \
    --candidate-run <path>       # Required: Path to candidate run
    --baseline-run <path>        # Optional: Path to baseline run
    --out-dir <path>             # Required: Output directory
    --alpha 0.05                 # Significance level (default: 0.05)
    --n-bootstrap 1000           # Bootstrap samples (default: 1000)
    --seed 42                    # Random seed (default: 42)
    --skip-evidence-verify       # Skip evidence verification
    --quiet                      # Suppress verbose output
```

### `smoke` Command

```bash
python3 -m batch_runs.phase_ab.cli smoke \
    --out-dir <path>             # Optional: Output directory
    --auto-generate-phasea       # Auto-generate Phase A runs if missing
    --alpha 0.05                 # Significance level (default: 0.05)
    --n-bootstrap 1000           # Bootstrap samples (default: 1000)
    --seed 42                    # Random seed (default: 42)
    --skip-evidence-verify       # Skip evidence verification
    --quiet                      # Suppress verbose output
```

## Programmatic Usage

```python
from pathlib import Path
from batch_runs.phase_ab import run_phase_ab

result = run_phase_ab(
    candidate_run=Path("runs/phaseA_candidate"),
    baseline_run=Path("runs/phaseA_baseline"),
    out_dir=Path("runs/phaseAB_output"),
    alpha=0.05,
    n_bootstrap=1000,
    seed=42,
    skip_evidence_verify=False,
    verbose=True,
)

print(f"Decision: {result.decision}")
print(f"Exit code: {result.exit_code}")
print(f"Manifest: {result.manifest.to_dict()}")
```

## Smoke Test Setup

The smoke test uses standard directories:

```
runs/
  phaseA_candidate_smoke/   <- Candidate run
    trials.jsonl
    pareto.json
    pareto.csv
    trial_*/...
  phaseA_baseline_smoke/    <- Baseline run
    trials.jsonl
    pareto.json
    pareto.csv
    trial_*/...
  phaseAB_smoke/            <- Output (created by smoke)
    phase_ab_manifest.json
    confidence_report.json
    confidence_report.md
```

Generate Phase A runs manually:

```bash
# Generate candidate
python3 -m batch_runs.phase_a.promote_pipeline \
    --smoke \
    --study-dir runs/phaseA_candidate_smoke \
    --seed 42

# Generate baseline
python3 -m batch_runs.phase_a.promote_pipeline \
    --smoke \
    --study-dir runs/phaseA_baseline_smoke \
    --seed 43
```

Or use `--auto-generate-phasea`:

```bash
python3 -m batch_runs.phase_ab.cli smoke --auto-generate-phasea
```

## Testing

Run unit tests:

```bash
python3 -m unittest discover -s batch_runs/phase_ab/tests -v
```

## Integration with CI

Phase AB is designed for CI pipelines:

1. **Deterministic**: Seeded RNG ensures reproducible results
2. **Exit codes**: Standard semantics for CI gates
3. **Artifacts**: All outputs are machine-readable
4. **Smoke mode**: Quick validation without path picking

Example GitHub Actions workflow:

```yaml
- name: Run PhaseAB smoke
  run: python3 -m batch_runs.phase_ab.cli smoke

- name: Upload artifacts
  uses: actions/upload-artifact@v4
  with:
    name: phaseAB-report
    path: runs/phaseAB_smoke/**/confidence_report.*
```

## Error Handling

Phase AB provides high-signal error messages:

| Error Type | Cause | Resolution |
|------------|-------|------------|
| `TrialsNotFoundError` | No `trials.jsonl` found | Check path, ensure Phase A completed |
| `AdversarialSearchError` | Directory is search output | Run `sim_eval suite` first |
| `NestedRunError` | Wrong nesting level | Use the suggested nested path |
| `EvidenceVerificationError` | Hash mismatch | Re-run Phase A or investigate tampering |

## Architecture

```
batch_runs/phase_ab/
  __init__.py          # Public API
  pipeline.py          # Core orchestration logic
  cli.py               # CLI wrapper
  tests/
    __init__.py
    test_phase_ab.py   # Unit tests
```

Phase AB imports and reuses:
- `batch_runs.phase_b.cli.run_gate` for gating logic
- `batch_runs.phase_b.gate.PromotionDecision` for decisions
- Evidence verification via `sim_eval verify-evidence-pack`

## Non-Negotiable Constraints

1. **No changes to Phase B decision logic**: Phase AB only orchestrates
2. **Deterministic outputs**: Same inputs → same outputs
3. **No commits to `runs/`**: Generated outputs are gitignored
4. **Fail-safe**: Invalid inputs fail with actionable errors, not silent corruption

