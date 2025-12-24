"""
Phase A: Promotion Pipeline + Multi-objective Tuning

This module implements the Phase A promotion pipeline per ROADMAP.md requirements:
- Multi-objective tuning of strategy config knobs
- Promotion gated by adversarial regression suites
- Evidence pack verification enforcement

Usage:
    python3 -m batch_runs.phase_a.cli optimize --trials 10 --mc-runs 30 --seed 42
    python3 -m batch_runs.phase_a.cli promote --study my_study
"""

__version__ = "0.1.0"

