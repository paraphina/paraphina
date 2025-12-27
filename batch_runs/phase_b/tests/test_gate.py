"""
Tests for Phase B promotion gate.

Tests:
- test_gate_rejection_reasoning: Clear rejection reasons
- test_gate_promotion_decision: Promotion decision structure
- test_gate_threshold_checks: Threshold-based gates
- test_tristate_decisions: PROMOTE/HOLD/REJECT decision semantics
"""

import math
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_b.confidence import compute_run_metrics
from batch_runs.phase_b.gate import (
    PromotionDecision,
    PromotionGate,
    PromotionOutcome,
)


def generate_pnl_data(
    n_runs: int = 20,
    mean_pnl: float = 100.0,
    std_pnl: float = 20.0,
    kill_rate: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Generate synthetic PnL data for testing."""
    np.random.seed(seed)
    
    pnl_per_run = []
    kill_flags = []
    
    for i in range(n_runs):
        # Generate a simple PnL series
        final_pnl = np.random.normal(mean_pnl, std_pnl)
        pnl_series = np.cumsum(np.random.randn(50) + final_pnl / 50)
        pnl_per_run.append(pnl_series)
        kill_flags.append(np.random.random() < kill_rate)
    
    return pnl_per_run, kill_flags


class TestPromotionGate(unittest.TestCase):
    """Tests for PromotionGate."""
    
    def test_gate_promotion_when_no_baseline(self):
        """Test gate promotes when thresholds are met and no baseline."""
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,  # Reduced for speed
            seed=42,
            kill_threshold=0.1,  # Candidate has 0% kills, should pass
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
        )
        
        self.assertEqual(decision.outcome, PromotionOutcome.PROMOTE)
        self.assertTrue(decision.is_promote)
        self.assertEqual(decision.exit_code, 0)
    
    def test_gate_rejection_when_kill_threshold_exceeded(self):
        """Test gate rejects when kill threshold is exceeded."""
        # All runs trigger kill switch
        candidate_pnl, _ = generate_pnl_data(n_runs=20)
        candidate_kills = [True] * 20  # 100% kill rate
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
            kill_threshold=0.1,  # Candidate has 100% kills, should fail
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
        )
        
        self.assertEqual(decision.outcome, PromotionOutcome.REJECT)
        self.assertTrue(decision.is_reject)
        self.assertEqual(decision.exit_code, 3)
        
        # Check that fail reason mentions kill rate
        kill_failures = [r for r in decision.fail_reasons if "kill" in r.lower()]
        self.assertGreater(len(kill_failures), 0)


class TestGateRejectionReasoning(unittest.TestCase):
    """Tests for clear rejection reasoning."""
    
    def test_gate_rejection_reasoning(self):
        """Test that rejection includes clear reasons."""
        # Generate candidate with high kill rate
        candidate_pnl, _ = generate_pnl_data(n_runs=20, seed=42)
        candidate_kills = [True] * 15 + [False] * 5  # 75% kill rate
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
            kill_threshold=0.1,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
        )
        
        self.assertEqual(decision.outcome, PromotionOutcome.REJECT)
        
        # Check that we have fail reasons
        self.assertGreater(len(decision.fail_reasons), 0)
        
        # Check that fail reasons are human-readable
        for reason in decision.fail_reasons:
            self.assertIsInstance(reason, str)
            self.assertGreater(len(reason), 0)
    
    def test_gate_provides_pass_reasons_on_promote(self):
        """Test that promotion includes pass reasons."""
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, kill_rate=0.0
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
            kill_threshold=0.2,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
        )
        
        self.assertEqual(decision.outcome, PromotionOutcome.PROMOTE)
        self.assertGreater(len(decision.pass_reasons), 0)
    
    def test_gate_baseline_comparison_rejection(self):
        """Test rejection when candidate is worse than baseline."""
        # Candidate with low PnL
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=20, mean_pnl=50.0, std_pnl=5.0, kill_rate=0.0, seed=42
        )
        
        # Baseline with high PnL (should dominate)
        baseline_pnl, baseline_kills = generate_pnl_data(
            n_runs=20, mean_pnl=200.0, std_pnl=5.0, kill_rate=0.0, seed=43
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
            require_strict_dominance=True,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
        )
        
        # Candidate should be rejected because baseline dominates
        self.assertEqual(decision.outcome, PromotionOutcome.REJECT)
        
        # Check that there are fail reasons about PnL or dominance
        self.assertGreater(len(decision.fail_reasons), 0)


class TestPromotionDecision(unittest.TestCase):
    """Tests for PromotionDecision structure."""
    
    def test_decision_exit_codes(self):
        """Test that exit codes are correct."""
        promote = PromotionDecision(
            outcome=PromotionOutcome.PROMOTE,
            candidate_path=None,
            baseline_path=None,
            candidate_metrics=None,
            baseline_metrics=None,
            comparison=None,
        )
        self.assertEqual(promote.exit_code, 0)
        
        hold = PromotionDecision(
            outcome=PromotionOutcome.HOLD,
            candidate_path=None,
            baseline_path=None,
            candidate_metrics=None,
            baseline_metrics=None,
            comparison=None,
        )
        self.assertEqual(hold.exit_code, 4)
        
        reject = PromotionDecision(
            outcome=PromotionOutcome.REJECT,
            candidate_path=None,
            baseline_path=None,
            candidate_metrics=None,
            baseline_metrics=None,
            comparison=None,
        )
        self.assertEqual(reject.exit_code, 3)
        
        error = PromotionDecision(
            outcome=PromotionOutcome.ERROR,
            candidate_path=None,
            baseline_path=None,
            candidate_metrics=None,
            baseline_metrics=None,
            comparison=None,
        )
        self.assertEqual(error.exit_code, 1)
    
    def test_decision_to_dict(self):
        """Test JSON serialization of decision."""
        decision = PromotionDecision(
            outcome=PromotionOutcome.PROMOTE,
            candidate_path=Path("/tmp/candidate"),
            baseline_path=None,
            candidate_metrics=None,
            baseline_metrics=None,
            comparison=None,
            pass_reasons=["Test passed"],
            fail_reasons=[],
        )
        
        d = decision.to_dict()
        
        self.assertEqual(d["outcome"], "promote")
        self.assertTrue(d["is_promote"])
        self.assertEqual(d["exit_code"], 0)
        self.assertIn("Test passed", d["pass_reasons"])
    
    def test_decision_summary(self):
        """Test that summary is human-readable."""
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=10, mean_pnl=100.0, kill_rate=0.0
        )
        
        metrics = compute_run_metrics(
            candidate_pnl, candidate_kills,
            alpha=0.05, n_bootstrap=50, seed=42
        )
        
        decision = PromotionDecision(
            outcome=PromotionOutcome.PROMOTE,
            candidate_path=Path("/tmp/candidate"),
            baseline_path=None,
            candidate_metrics=metrics,
            baseline_metrics=None,
            comparison=None,
            pass_reasons=["All gates passed"],
        )
        
        summary = decision.summary()
        
        self.assertIn("PROMOTE", summary)
        self.assertIn("Candidate Metrics", summary)
        self.assertIn("PnL mean", summary)
    
    def test_decision_markdown(self):
        """Test Markdown report generation."""
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=10, mean_pnl=100.0, kill_rate=0.0
        )
        
        metrics = compute_run_metrics(
            candidate_pnl, candidate_kills,
            alpha=0.05, n_bootstrap=50, seed=42
        )
        
        decision = PromotionDecision(
            outcome=PromotionOutcome.REJECT,
            candidate_path=Path("/tmp/candidate"),
            baseline_path=None,
            candidate_metrics=metrics,
            baseline_metrics=None,
            comparison=None,
            guardrail_checks=[("Kill rate threshold", False, "Kill rate 0.50 > threshold 0.10")],
            guardrails_passed=False,
            promotion_passed=False,
            decision_reason="REJECT: Guardrails failed (Kill rate threshold). Candidate is provably worse.",
            fail_reasons=["[Guardrail] Kill rate threshold: Kill rate 0.50 > threshold 0.10"],
        )
        
        md = decision.to_markdown()
        
        self.assertIn("# Phase B", md)
        self.assertIn("REJECT", md)
        self.assertIn("Kill rate", md)
        self.assertIn("Guardrails (must pass)", md)


class TestGateDeterminism(unittest.TestCase):
    """Tests for deterministic gate behavior."""
    
    def test_gate_is_deterministic(self):
        """Test that gate gives same result with same seed."""
        candidate_pnl, candidate_kills = generate_pnl_data(n_runs=15, seed=42)
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision1 = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
        )
        
        decision2 = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
        )
        
        self.assertEqual(decision1.outcome, decision2.outcome)
        self.assertEqual(
            decision1.candidate_metrics.pnl_mean.lower,
            decision2.candidate_metrics.pnl_mean.lower,
        )


class TestTriStateDecisions(unittest.TestCase):
    """Tests for tri-state decision model: PROMOTE, HOLD, REJECT."""
    
    def test_identical_metrics_yields_hold(self):
        """Test that candidate == baseline (identical metrics) yields HOLD."""
        # Generate identical data for both candidate and baseline
        pnl_data, kill_flags = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=42
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision = gate.evaluate(
            candidate_pnl=pnl_data,
            candidate_kills=kill_flags,
            baseline_pnl=pnl_data,  # Same data!
            baseline_kills=kill_flags,
        )
        
        # With identical data, guardrails pass but promotion criteria fail
        # (CIs overlap, so candidate is not provably better)
        self.assertEqual(decision.outcome, PromotionOutcome.HOLD)
        self.assertTrue(decision.is_hold)
        self.assertEqual(decision.exit_code, 4)
        
        # Guardrails should pass
        self.assertTrue(decision.guardrails_passed)
        # Promotion criteria should fail (CIs overlap)
        self.assertFalse(decision.promotion_passed)
        
        # Decision reason should explain HOLD
        self.assertIn("HOLD", decision.decision_reason)
    
    def test_candidate_provably_worse_yields_reject(self):
        """Test that candidate provably worse than baseline yields REJECT."""
        # Candidate with much lower PnL (should be provably worse)
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=30, mean_pnl=20.0, std_pnl=3.0, kill_rate=0.0, seed=42
        )
        
        # Baseline with much higher PnL (clearly dominates)
        baseline_pnl, baseline_kills = generate_pnl_data(
            n_runs=30, mean_pnl=200.0, std_pnl=3.0, kill_rate=0.0, seed=43
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
        )
        
        # Baseline dominates, so guardrails fail → REJECT
        self.assertEqual(decision.outcome, PromotionOutcome.REJECT)
        self.assertTrue(decision.is_reject)
        self.assertEqual(decision.exit_code, 3)
        
        # Guardrails should fail
        self.assertFalse(decision.guardrails_passed)
        
        # Decision reason should explain REJECT
        self.assertIn("REJECT", decision.decision_reason)
        self.assertIn("Guardrails failed", decision.decision_reason)
    
    def test_candidate_provably_better_on_pnl_with_equal_drawdown_yields_hold(self):
        """Test that candidate better on PnL but equal drawdown yields HOLD.
        
        When candidate is provably better on PnL but drawdowns are equal (both 0),
        the drawdown superiority check fails (CIs overlap), resulting in HOLD.
        This is the expected behavior for smoke runs.
        """
        # Candidate with much higher PnL
        np.random.seed(42)
        candidate_pnl = [np.array([300.0 + np.random.randn() * 2]) for _ in range(30)]
        candidate_kills = [False] * 30
        
        # Baseline with much lower PnL
        np.random.seed(43)
        baseline_pnl = [np.array([50.0 + np.random.randn() * 2]) for _ in range(30)]
        baseline_kills = [False] * 30
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
        )
        
        # Guardrails should pass (candidate not worse)
        self.assertTrue(decision.guardrails_passed, 
            f"Expected guardrails to pass. Checks: {decision.guardrail_checks}")
        
        # Drawdowns are both 0, so CI overlap causes drawdown superiority to fail
        # Result is HOLD (guardrails pass but promotion criteria fail)
        self.assertEqual(decision.outcome, PromotionOutcome.HOLD)
        self.assertIn("HOLD", decision.decision_reason)
    
    def test_candidate_dominates_all_metrics_yields_promote(self):
        """Test that candidate dominating ALL metrics yields PROMOTE.
        
        For PROMOTE, candidate must be provably better on BOTH PnL and drawdown.
        """
        # Candidate: high PnL, low drawdown (no drops in equity curve)
        np.random.seed(42)
        candidate_pnl = []
        for _ in range(30):
            # Strictly positive increments -> 0 drawdown, high final PnL
            series = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])  # equity = [50, 100, 150...] no drawdown
            candidate_pnl.append(series)
        candidate_kills = [False] * 30
        
        # Baseline: low PnL, higher drawdown (has drops in equity curve)
        np.random.seed(43)
        baseline_pnl = []
        for _ in range(30):
            # Equity curve that goes up, drops, recovers - creates drawdown
            # pnl series: [10, -20, 15, -10, 5] -> equity [10, -10, 5, -5, 0]
            # Running max: [10, 10, 10, 10, 10], drawdown = [0, 20, 5, 15, 10], max = 20
            series = np.array([10.0, -20.0, 15.0, -10.0, 5.0])
            baseline_pnl.append(series)
        baseline_kills = [False] * 30
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
        )
        
        # Guardrails should pass
        self.assertTrue(decision.guardrails_passed,
            f"Expected guardrails to pass. Checks: {decision.guardrail_checks}")
        
        # PnL: candidate (300) >> baseline (0) -> candidate dominates
        # Drawdown: candidate (0) < baseline (~20) -> candidate dominates
        # Both promotion criteria should pass -> PROMOTE
        self.assertTrue(decision.promotion_passed,
            f"Expected promotion to pass. Checks: {decision.promotion_checks}")
        
        self.assertEqual(decision.outcome, PromotionOutcome.PROMOTE)
        self.assertTrue(decision.is_promote)
        self.assertEqual(decision.exit_code, 0)
        self.assertIn("PROMOTE", decision.decision_reason)
    
    def test_decision_includes_guardrail_checks(self):
        """Test that decision includes guardrail_checks list."""
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=42
        )
        baseline_pnl, baseline_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=43
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
        )
        
        # Should have guardrail checks
        self.assertGreater(len(decision.guardrail_checks), 0)
        
        # Each check should be (name, passed, reason) tuple
        for check in decision.guardrail_checks:
            self.assertEqual(len(check), 3)
            name, passed, reason = check
            self.assertIsInstance(name, str)
            self.assertIsInstance(passed, bool)
            self.assertIsInstance(reason, str)
    
    def test_decision_includes_promotion_checks(self):
        """Test that decision includes promotion_checks list."""
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=42
        )
        baseline_pnl, baseline_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=43
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
        )
        
        # Should have promotion checks
        self.assertGreater(len(decision.promotion_checks), 0)
        
        # Each check should be (name, passed, reason) tuple
        for check in decision.promotion_checks:
            self.assertEqual(len(check), 3)
            name, passed, reason = check
            self.assertIsInstance(name, str)
            self.assertIsInstance(passed, bool)
            self.assertIsInstance(reason, str)
    
    def test_json_includes_new_fields(self):
        """Test that to_dict() includes new tri-state fields."""
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=42
        )
        baseline_pnl, baseline_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=43
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
        )
        
        d = decision.to_dict()
        
        # Check new fields exist
        self.assertIn("decision", d)
        self.assertIn("guardrails_passed", d)
        self.assertIn("promotion_passed", d)
        self.assertIn("decision_reason", d)
        self.assertIn("guardrail_checks", d)
        self.assertIn("promotion_checks", d)
        
        # Check decision field matches outcome
        self.assertEqual(d["decision"], decision.outcome.value.upper())
    
    def test_summary_has_new_sections(self):
        """Test that summary() includes new guardrail/promotion sections."""
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=42
        )
        baseline_pnl, baseline_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=43
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
        )
        
        summary = decision.summary()
        
        # Check for new section headings
        self.assertIn("Guardrails (must pass):", summary)
        self.assertIn("Promotion criteria (must pass to PROMOTE):", summary)
        self.assertIn("Decision rationale:", summary)
    
    def test_markdown_has_new_sections(self):
        """Test that to_markdown() includes new sections."""
        candidate_pnl, candidate_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=42
        )
        baseline_pnl, baseline_kills = generate_pnl_data(
            n_runs=20, mean_pnl=100.0, std_pnl=10.0, kill_rate=0.0, seed=43
        )
        
        gate = PromotionGate(
            alpha=0.05,
            n_bootstrap=100,
            seed=42,
        )
        
        decision = gate.evaluate(
            candidate_pnl=candidate_pnl,
            candidate_kills=candidate_kills,
            baseline_pnl=baseline_pnl,
            baseline_kills=baseline_kills,
        )
        
        md = decision.to_markdown()
        
        # Check for new section headings
        self.assertIn("## Decision Rationale", md)
        self.assertIn("## Guardrails (must pass)", md)
        self.assertIn("## Promotion Criteria (must pass to PROMOTE)", md)
        self.assertIn("Guardrails passed:", md)
        self.assertIn("Promotion criteria passed:", md)


if __name__ == "__main__":
    unittest.main()

