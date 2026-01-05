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

# CI-friendly with deterministic output and seed
python3 -m batch_runs.phase_ab.cli smoke --auto-generate-phasea \
    --out-dir runs/ci/phase_ab_smoke --seed 12345
```

### Promotion Gate Mode (Strict/Institutional)

```bash
# Promotion gate with auto-generate (for CI)
python3 -m batch_runs.phase_ab.cli gate \
    --auto-generate-phasea \
    --out-dir runs/ci/phase_ab_gate \
    --seed 24680 \
    --n-bootstrap 1000

# Promotion gate with explicit paths
python3 -m batch_runs.phase_ab.cli gate \
    --candidate-run runs/phaseA_candidate \
    --baseline-run runs/phaseA_baseline \
    --out-dir runs/ci/phase_ab_gate \
    --seed 24680
```

### Verify Evidence Pack

```bash
# Verify an evidence pack after download
python3 -m batch_runs.phase_ab.cli verify-evidence runs/ci/phase_ab_smoke/evidence_pack
```

## When to Use Smoke vs Gate

| Mode | Use Case | HOLD Behavior | Exit Codes |
|------|----------|---------------|------------|
| **smoke** | CI smoke tests, integration tests, development | HOLD = exit 0 (CI pass) | 0=PASS/HOLD, 2=REJECT, 3=ERROR |
| **gate** | Production promotion decisions, institutional gates | HOLD = exit 2 (needs more data) | 0=PASS, 1=FAIL, 2=HOLD, 3=ERROR |

- Use **smoke** to verify the pipeline runs correctly without blocking on statistical significance
- Use **gate** when you require definitive statistical evidence before promotion

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
| `phase_ab_summary.json` | Machine-readable CI summary |
| `evidence_pack/manifest.json` | Evidence pack integrity manifest |

### 5. Evidence Pack Manifest

When `--out-dir` is provided, Phase AB:
1. Creates an `evidence_pack/` subdirectory
2. Copies key output files to the evidence pack
3. Generates a deterministic `manifest.json` with SHA-256 hashes
4. Self-verifies the manifest immediately after creation

The manifest enables integrity verification:

```bash
python3 -m batch_runs.phase_ab.cli verify-evidence runs/ci/phase_ab_smoke/evidence_pack
```

See [Evidence Pack Manifest](./evidence_pack_manifest.md) for details.

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

Phase AB uses institutional CI exit code semantics with two modes:

### CI Mode: Smoke (Default for `smoke` command)

Used for smoke tests and integration tests where we verify the pipeline runs correctly.

| Exit Code | Decision | Meaning |
|-----------|----------|---------|
| 0 | PROMOTE | Candidate is provably better |
| 0 | HOLD | Pipeline succeeded; not enough evidence yet |
| 2 | REJECT | Candidate fails guardrails |
| 3 | ERROR | Runtime/IO/parsing failure |

**Key insight:** HOLD is exit code 0 in smoke mode because the pipeline *succeeded*—there's just not enough evidence to promote. This is expected for smoke runs with few trials.

### CI Mode: Strict/Gate (Default for `gate` command)

Used for promotion gates that require definitive superiority. The `gate` command always uses strict mode.

| Exit Code | Decision | Meaning |
|-----------|----------|---------|
| 0 | PROMOTE | PASS - Candidate is provably better |
| 1 | REJECT | FAIL - Candidate fails guardrails |
| 2 | HOLD | Insufficient evidence - needs more data |
| 3 | ERROR | Runtime/IO/parsing failure |

**Key difference from smoke mode:**
- REJECT returns exit code 1 (not 2) - clear FAIL signal
- HOLD returns exit code 2 (not 0) - requires explicit action (collect more data)
- Exit codes 0, 1, 2, 3 are distinct and deterministic

### Choosing CI Mode

```bash
# Smoke mode (default for smoke command): HOLD is CI pass
python3 -m batch_runs.phase_ab.cli smoke --auto-generate-phasea ...

# Gate mode (promotion gate with mandatory evidence verification)
python3 -m batch_runs.phase_ab.cli gate --out-dir ... --seed ...

# Strict mode with run command: HOLD returns exit 2
python3 -m batch_runs.phase_ab.cli run --ci-mode strict ...
```

**When to use each mode:**
- **smoke**: For CI smoke tests, integration tests, and development validation
- **gate**: For production promotion gates with mandatory evidence verification
- **run --ci-mode strict**: For one-off promotion checks without auto-generation

## Evidence Verification in Gate Mode

The `gate` command enforces mandatory evidence verification:

1. Runs Phase AB evaluation in strict mode
2. Writes outputs to `--out-dir`
3. Creates evidence pack with SHA-256 hashes
4. **Immediately verifies** the evidence pack (cannot be skipped)
5. Returns deterministic exit code

If evidence verification fails, gate returns exit code 3 (ERROR).

```bash
# Gate command enforces evidence verification automatically
python3 -m batch_runs.phase_ab.cli gate \
    --auto-generate-phasea \
    --out-dir runs/ci/phase_ab_gate \
    --seed 24680
```

## What HOLD Means

A **HOLD** decision indicates:

1. ✅ All guardrails passed (candidate not provably worse)
2. ⚠ Promotion criteria failed (confidence intervals overlap)
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
    --ci-mode smoke|strict       # CI mode (default: smoke)
    --quiet                      # Suppress verbose output
```

### `smoke` Command

```bash
python3 -m batch_runs.phase_ab.cli smoke \
    --out-dir <path>             # Optional: Output directory (default: runs/phaseAB_smoke)
    --auto-generate-phasea       # Auto-generate Phase A runs if missing
    --alpha 0.05                 # Significance level (default: 0.05)
    --n-bootstrap 1000           # Bootstrap samples (default: 1000)
    --seed 42                    # Random seed for reproducibility (default: 42)
    --skip-evidence-verify       # Skip evidence verification
    --ci-mode smoke|strict       # CI mode (default: smoke)
    --quiet                      # Suppress verbose output
```

### `gate` Command

```bash
python3 -m batch_runs.phase_ab.cli gate \
    --out-dir <path>             # Required: Output directory
    --seed <int>                 # Required: Random seed for reproducibility
    --n-bootstrap 1000           # Bootstrap samples (default: 1000)
    --candidate-run <path>       # Path to candidate run (required if not auto-generate)
    --baseline-run <path>        # Path to baseline run (optional)
    --auto-generate-phasea       # Auto-generate Phase A runs if missing
    --alpha 0.05                 # Significance level (default: 0.05)
    --quiet                      # Suppress verbose output
```

Exit codes (strict mode - deterministic):
- `0`: PASS (PROMOTE - candidate is provably better)
- `1`: FAIL (REJECT - candidate fails guardrails)
- `2`: HOLD (insufficient evidence - needs more data)
- `3`: ERROR (runtime/IO/parsing failure)

**Note:** The `gate` command:
- Always uses strict mode (cannot be changed)
- Mandatory evidence verification (cannot be skipped)
- Requires `--out-dir` and `--seed` (for determinism)

### `verify-evidence` Command

```bash
python3 -m batch_runs.phase_ab.cli verify-evidence <EVIDENCE_PACK_DIR> \
    --allow-extra                # Allow extra files not in manifest
```

Exit codes:
- `0`: Verification passed
- `2`: Verification failed

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
3. **Artifacts**: All outputs are machine-readable with integrity guarantees
4. **Smoke mode**: Quick validation without path picking
5. **Gate mode**: Institutional-grade promotion with mandatory evidence verification
6. **Evidence pack**: Verifiable artifact with SHA-256 hashes

### GitHub Workflows

#### Smoke Workflow (`phase_ab_smoke.yml`)

Runs automatically on PRs and pushes to main. Uses smoke mode (HOLD = exit 0).

```bash
# Triggered automatically on PR/push
# Uses --ci-mode smoke (default)
```

Artifacts:
- **phase-ab-smoke-artifacts**: JSON/MD outputs
- **phase-ab-evidence-pack**: Evidence pack with manifest

#### Promotion Gate Workflow (`phase_ab_promotion_gate.yml`)

Manually triggered via GitHub UI. Uses strict/gate mode with deterministic exit codes.

```bash
# Trigger from GitHub UI: Actions → phase-ab-promotion-gate → Run workflow
```

**How to run from GitHub UI:**
1. Go to Actions tab
2. Select "phase-ab-promotion-gate" workflow
3. Click "Run workflow"
4. Optionally configure seed and n_bootstrap
5. Click "Run workflow" button

**Exit codes:**
- 0 = PASS (PROMOTE)
- 1 = FAIL (REJECT)
- 2 = HOLD (needs more data)
- 3 = ERROR

Artifacts:
- **phase-ab-gate-artifacts**: JSON/MD outputs (manifest, confidence reports, summary)
- **phase-ab-gate-evidence-pack**: Cryptographically verifiable evidence pack

### CI Artifact Upload Examples

#### Smoke mode

```yaml
- name: Run PhaseAB smoke with deterministic output
  run: |
    python3 -m batch_runs.phase_ab.cli smoke \
      --auto-generate-phasea \
      --out-dir runs/ci/phase_ab_smoke \
      --seed 12345

- name: Upload evidence pack artifact
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: phase-ab-evidence-pack
    path: runs/ci/phase_ab_smoke
```

#### Gate mode

```yaml
- name: Run PhaseAB promotion gate
  run: |
    python3 -m batch_runs.phase_ab.cli gate \
      --auto-generate-phasea \
      --out-dir runs/ci/phase_ab_gate \
      --seed 24680 \
      --n-bootstrap 1000

- name: Upload gate artifacts
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: phase-ab-gate-artifacts
    path: |
      runs/ci/phase_ab_gate/phase_ab_manifest.json
      runs/ci/phase_ab_gate/confidence_report.json
      runs/ci/phase_ab_gate/confidence_report.md

- name: Upload evidence pack
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: phase-ab-gate-evidence-pack
    path: runs/ci/phase_ab_gate/evidence_pack
```

### Verifying Downloaded Artifacts

After downloading the artifact from GitHub Actions:

```bash
unzip phase-ab-evidence-pack.zip -d artifact
python3 -m batch_runs.phase_ab.cli verify-evidence artifact/evidence_pack
```

### Summary Outputs

Phase AB writes `phase_ab_summary.json` with:
- `outcome`: PASS/FAIL/HOLD
- `mode`: "smoke" or "run"
- `evidence_pack_dir`: Path to evidence pack
- `manifest_path`: Path to manifest.json
- `seed`: Random seed (if provided)
- `git_commit`: Git commit hash
- `python_version`: Python version

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

