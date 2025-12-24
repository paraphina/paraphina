"""
Tests for deterministic winner selection tie-breaks.

Hermetic tests - no cargo invocation.
"""

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.schemas import (
    TierBudget,
    CandidateConfig,
    EvalResult,
)
from batch_runs.phase_a.optimize import select_winner_deterministic


class TestWinnerSelection(unittest.TestCase):
    """Test deterministic winner selection with tie-breaks."""
    
    def setUp(self):
        """Create test fixtures."""
        self.budget = TierBudget(
            tier_name="balanced",
            max_kill_prob=0.10,
            max_drawdown_cvar=1000.0,
            min_mean_pnl=20.0,
        )
    
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
    
    def test_no_qualifying_candidates(self):
        """Test when no candidates meet budget."""
        results = [
            self._make_result("c1", mean_pnl=10.0, kill_prob=0.05, drawdown_cvar=500.0),
            self._make_result("c2", mean_pnl=15.0, kill_prob=0.15, drawdown_cvar=500.0),
        ]
        
        winner = select_winner_deterministic(results, self.budget)
        
        self.assertIsNone(winner)
    
    def test_single_qualifying_candidate(self):
        """Test with single qualifying candidate."""
        results = [
            self._make_result("c1", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),
            self._make_result("c2", mean_pnl=10.0, kill_prob=0.05, drawdown_cvar=500.0),  # Low pnl
        ]
        
        winner = select_winner_deterministic(results, self.budget)
        
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_id, "c1")
    
    def test_select_highest_pnl(self):
        """Test that highest mean_pnl is selected."""
        results = [
            self._make_result("c1", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),
            self._make_result("c2", mean_pnl=80.0, kill_prob=0.05, drawdown_cvar=500.0),
            self._make_result("c3", mean_pnl=30.0, kill_prob=0.05, drawdown_cvar=500.0),
        ]
        
        winner = select_winner_deterministic(results, self.budget)
        
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_id, "c2")
    
    def test_tie_break_by_drawdown(self):
        """Test tie-break by drawdown_cvar when pnl is equal."""
        results = [
            self._make_result("c1", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=800.0),
            self._make_result("c2", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=400.0),
            self._make_result("c3", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=600.0),
        ]
        
        winner = select_winner_deterministic(results, self.budget)
        
        # c2 has lowest drawdown_cvar
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_id, "c2")
    
    def test_tie_break_by_kill_prob(self):
        """Test tie-break by kill_prob when pnl and drawdown are equal."""
        results = [
            self._make_result("c1", mean_pnl=50.0, kill_prob=0.08, drawdown_cvar=500.0),
            self._make_result("c2", mean_pnl=50.0, kill_prob=0.03, drawdown_cvar=500.0),
            self._make_result("c3", mean_pnl=50.0, kill_prob=0.06, drawdown_cvar=500.0),
        ]
        
        winner = select_winner_deterministic(results, self.budget)
        
        # c2 has lowest kill_prob
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_id, "c2")
    
    def test_tie_break_by_candidate_id(self):
        """Test final tie-break by candidate_id for determinism."""
        results = [
            self._make_result("c_charlie", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),
            self._make_result("c_alpha", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),
            self._make_result("c_bravo", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),
        ]
        
        winner = select_winner_deterministic(results, self.budget)
        
        # c_alpha comes first alphabetically
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_id, "c_alpha")
    
    def test_determinism_multiple_runs(self):
        """Test that winner selection is deterministic across multiple runs."""
        results = [
            self._make_result("c1", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),
            self._make_result("c2", mean_pnl=60.0, kill_prob=0.07, drawdown_cvar=600.0),
            self._make_result("c3", mean_pnl=55.0, kill_prob=0.06, drawdown_cvar=550.0),
        ]
        
        # Run multiple times
        winners = []
        for _ in range(10):
            winner = select_winner_deterministic(results, self.budget)
            winners.append(winner.candidate_id if winner else None)
        
        # All should be the same
        self.assertEqual(len(set(winners)), 1)
    
    def test_invalid_candidates_excluded(self):
        """Test that candidates with failed evidence verification are excluded."""
        results = [
            # Best pnl but invalid evidence
            self._make_result(
                "c1", mean_pnl=100.0, kill_prob=0.05, drawdown_cvar=400.0,
                evidence_verified=False
            ),
            # Lower pnl but valid
            self._make_result(
                "c2", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0,
                evidence_verified=True
            ),
        ]
        
        winner = select_winner_deterministic(results, self.budget)
        
        # c2 should win despite lower pnl because c1 is invalid
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_id, "c2")


if __name__ == "__main__":
    unittest.main()

