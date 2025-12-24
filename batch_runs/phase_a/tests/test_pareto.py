"""
Tests for Pareto frontier computation.

Hermetic tests - no cargo invocation.
"""

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.schemas import (
    CandidateConfig,
    EvalResult,
    ParetoCandidate,
)
from batch_runs.phase_a.optimize import compute_pareto_frontier


class TestParetoFrontier(unittest.TestCase):
    """Test Pareto frontier computation."""
    
    def _make_result(
        self,
        candidate_id: str,
        mean_pnl: float,
        kill_prob: float,
        drawdown_cvar: float,
        evidence_verified: bool = True,
    ) -> EvalResult:
        """Helper to create test EvalResult."""
        config = CandidateConfig(
            candidate_id=candidate_id,
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
        
        return EvalResult(
            candidate_id=candidate_id,
            trial_id=f"trial_{candidate_id}",
            config=config,
            trial_dir=Path("/tmp/test"),
            research_mean_pnl=mean_pnl,
            research_kill_prob=kill_prob,
            research_drawdown_cvar=drawdown_cvar,
            adversarial_mean_pnl=mean_pnl,
            adversarial_kill_prob=kill_prob,
            adversarial_drawdown_cvar=drawdown_cvar,
            mc_mean_pnl=mean_pnl,
            mc_pnl_cvar=-100.0,
            mc_drawdown_cvar=drawdown_cvar,
            mc_kill_prob_point=kill_prob,
            mc_kill_prob_ci_upper=kill_prob,
            mc_total_runs=50,
            evidence_verified=evidence_verified,
        )
    
    def test_empty_input(self):
        """Test Pareto frontier of empty list."""
        pareto = compute_pareto_frontier([])
        self.assertEqual(len(pareto), 0)
    
    def test_single_candidate(self):
        """Test Pareto frontier with single valid candidate."""
        results = [
            self._make_result("c1", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),
        ]
        
        pareto = compute_pareto_frontier(results)
        
        self.assertEqual(len(pareto), 1)
        self.assertEqual(pareto[0].candidate_id, "c1")
    
    def test_clear_dominance(self):
        """Test case where one candidate clearly dominates another."""
        results = [
            # c1 dominates c2 on all objectives
            self._make_result("c1", mean_pnl=100.0, kill_prob=0.05, drawdown_cvar=400.0),
            self._make_result("c2", mean_pnl=50.0, kill_prob=0.10, drawdown_cvar=800.0),
        ]
        
        pareto = compute_pareto_frontier(results)
        
        self.assertEqual(len(pareto), 1)
        self.assertEqual(pareto[0].candidate_id, "c1")
    
    def test_no_dominance_both_on_frontier(self):
        """Test case where neither candidate dominates the other."""
        results = [
            # c1: higher pnl but higher kill_prob
            self._make_result("c1", mean_pnl=100.0, kill_prob=0.10, drawdown_cvar=500.0),
            # c2: lower pnl but lower kill_prob
            self._make_result("c2", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),
        ]
        
        pareto = compute_pareto_frontier(results)
        
        self.assertEqual(len(pareto), 2)
        ids = {r.candidate_id for r in pareto}
        self.assertIn("c1", ids)
        self.assertIn("c2", ids)
    
    def test_multi_candidate_pareto_frontier(self):
        """Test Pareto frontier with multiple candidates."""
        results = [
            # On frontier: high pnl, high risk
            self._make_result("c1", mean_pnl=100.0, kill_prob=0.15, drawdown_cvar=600.0),
            # On frontier: balanced
            self._make_result("c2", mean_pnl=70.0, kill_prob=0.08, drawdown_cvar=400.0),
            # On frontier: low risk, low pnl
            self._make_result("c3", mean_pnl=40.0, kill_prob=0.02, drawdown_cvar=200.0),
            # Dominated by c2
            self._make_result("c4", mean_pnl=60.0, kill_prob=0.10, drawdown_cvar=500.0),
            # Dominated by c1
            self._make_result("c5", mean_pnl=80.0, kill_prob=0.20, drawdown_cvar=800.0),
        ]
        
        pareto = compute_pareto_frontier(results)
        
        self.assertEqual(len(pareto), 3)
        ids = {r.candidate_id for r in pareto}
        self.assertIn("c1", ids)
        self.assertIn("c2", ids)
        self.assertIn("c3", ids)
        self.assertNotIn("c4", ids)
        self.assertNotIn("c5", ids)
    
    def test_invalid_candidates_excluded(self):
        """Test that invalid candidates are excluded from Pareto frontier."""
        results = [
            self._make_result("c1", mean_pnl=100.0, kill_prob=0.05, drawdown_cvar=400.0),
            self._make_result(
                "c2", mean_pnl=150.0, kill_prob=0.03, drawdown_cvar=300.0,
                evidence_verified=False  # Invalid
            ),
        ]
        
        pareto = compute_pareto_frontier(results)
        
        # c2 would dominate c1 but it's invalid
        self.assertEqual(len(pareto), 1)
        self.assertEqual(pareto[0].candidate_id, "c1")
    
    def test_deterministic_ordering(self):
        """Test that Pareto frontier is deterministically ordered."""
        results = [
            self._make_result("c3", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),
            self._make_result("c1", mean_pnl=50.0, kill_prob=0.10, drawdown_cvar=300.0),
            self._make_result("c2", mean_pnl=50.0, kill_prob=0.08, drawdown_cvar=400.0),
        ]
        
        # Run multiple times to verify determinism
        for _ in range(5):
            pareto = compute_pareto_frontier(results)
            
            # Should be sorted by candidate_id
            ids = [r.candidate_id for r in pareto]
            self.assertEqual(ids, sorted(ids))


class TestParetoCandidate(unittest.TestCase):
    """Test ParetoCandidate dominance logic."""
    
    def _make_pareto_candidate(
        self,
        mean_pnl: float,
        kill_prob: float,
        drawdown_cvar: float,
    ) -> ParetoCandidate:
        """Helper to create test ParetoCandidate."""
        config = CandidateConfig(
            candidate_id="test",
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
        
        result = EvalResult(
            candidate_id="test",
            trial_id="trial_test",
            config=config,
            trial_dir=Path("/tmp/test"),
            research_mean_pnl=mean_pnl,
            research_kill_prob=kill_prob,
            research_drawdown_cvar=drawdown_cvar,
            adversarial_mean_pnl=mean_pnl,
            adversarial_kill_prob=kill_prob,
            adversarial_drawdown_cvar=drawdown_cvar,
            mc_mean_pnl=mean_pnl,
            mc_pnl_cvar=-100.0,
            mc_drawdown_cvar=drawdown_cvar,
            mc_kill_prob_point=kill_prob,
            mc_kill_prob_ci_upper=kill_prob,
            mc_total_runs=50,
            evidence_verified=True,
        )
        
        return ParetoCandidate.from_result(result)
    
    def test_dominates_all_better(self):
        """Test dominance when strictly better on all objectives."""
        a = self._make_pareto_candidate(mean_pnl=100.0, kill_prob=0.05, drawdown_cvar=400.0)
        b = self._make_pareto_candidate(mean_pnl=50.0, kill_prob=0.10, drawdown_cvar=800.0)
        
        self.assertTrue(a.dominates(b))
        self.assertFalse(b.dominates(a))
    
    def test_dominates_equal_and_one_better(self):
        """Test dominance when equal on some, better on one."""
        a = self._make_pareto_candidate(mean_pnl=100.0, kill_prob=0.10, drawdown_cvar=500.0)
        b = self._make_pareto_candidate(mean_pnl=50.0, kill_prob=0.10, drawdown_cvar=500.0)
        
        # a dominates b: equal on kill_prob and drawdown, better on pnl
        self.assertTrue(a.dominates(b))
        self.assertFalse(b.dominates(a))
    
    def test_no_dominance_tradeoff(self):
        """Test no dominance when there's a tradeoff."""
        a = self._make_pareto_candidate(mean_pnl=100.0, kill_prob=0.10, drawdown_cvar=500.0)
        b = self._make_pareto_candidate(mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0)
        
        # Neither dominates: a has better pnl, b has better kill_prob
        self.assertFalse(a.dominates(b))
        self.assertFalse(b.dominates(a))
    
    def test_no_self_dominance(self):
        """Test that a candidate doesn't dominate itself."""
        a = self._make_pareto_candidate(mean_pnl=100.0, kill_prob=0.05, drawdown_cvar=400.0)
        
        # Can't be strictly better on at least one if identical
        self.assertFalse(a.dominates(a))


if __name__ == "__main__":
    unittest.main()

