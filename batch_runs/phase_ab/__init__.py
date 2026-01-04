"""
Phase AB: Integrated Phase A → Phase B Orchestrator.

This module provides institutional-grade orchestration for running Phase B
confidence-aware statistical gating on Phase A run outputs.

Usage:
    # CLI
    python3 -m batch_runs.phase_ab.cli run \\
        --candidate-run runs/phaseA_candidate \\
        --baseline-run runs/phaseA_baseline \\
        --out-dir runs/phaseAB_output

    # Smoke test (CI-friendly)
    python3 -m batch_runs.phase_ab.cli smoke

    # Programmatic
    from batch_runs.phase_ab import run_phase_ab
    result = run_phase_ab(
        candidate_run=Path("runs/phaseA_candidate"),
        baseline_run=Path("runs/phaseA_baseline"),
        out_dir=Path("runs/phaseAB_output"),
    )

Features:
- Robust run-root validation with trials.jsonl discovery
- Evidence pack verification (optional)
- Phase B confidence gating integration
- Canonical manifest output (phase_ab_manifest.json)
- CI-friendly smoke entrypoint

Exit codes (institutional CI semantics):
- 0: PROMOTE or HOLD (pipeline succeeded)
- 2: REJECT (candidate fails guardrails)
- 3: ERROR (runtime/IO/parsing failure)
"""

from batch_runs.phase_ab.pipeline import (
    run_phase_ab,
    resolve_run_root,
    verify_evidence_pack,
    PhaseABResult,
    PhaseABManifest,
    RunRootError,
    AdversarialSearchError,
    NestedRunError,
    TrialsNotFoundError,
)

__all__ = [
    "run_phase_ab",
    "resolve_run_root",
    "verify_evidence_pack",
    "PhaseABResult",
    "PhaseABManifest",
    "RunRootError",
    "AdversarialSearchError",
    "NestedRunError",
    "TrialsNotFoundError",
]

