"""
Tests for budget filtering logic.

Hermetic tests - no cargo invocation.
"""

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.schemas import (
    TierBudget,
    BudgetConfig,
    CandidateConfig,
    EvalResult,
)
from batch_runs.phase_a.optimize import (
    filter_by_budget,
    load_default_budgets,
)


class TestBudgetFiltering(unittest.TestCase):
    """Test budget filtering logic."""
    
    def setUp(self):
        """Create test fixtures."""
        self.balanced_budget = TierBudget(
            tier_name="balanced",
            max_kill_prob=0.10,
            max_drawdown_cvar=1000.0,
            min_mean_pnl=20.0,
        )
        
        self.conservative_budget = TierBudget(
            tier_name="conservative",
            max_kill_prob=0.05,
            max_drawdown_cvar=500.0,
            min_mean_pnl=10.0,
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
    
    def test_passes_budget_all_constraints(self):
        """Test that result passes when all constraints are met."""
        result = self._make_result("c1", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0)
        self.assertTrue(result.passes_budget(self.balanced_budget))
    
    def test_fails_budget_high_kill_prob(self):
        """Test that result fails when kill_prob exceeds budget."""
        result = self._make_result("c1", mean_pnl=50.0, kill_prob=0.15, drawdown_cvar=500.0)
        self.assertFalse(result.passes_budget(self.balanced_budget))
    
    def test_fails_budget_high_drawdown(self):
        """Test that result fails when drawdown_cvar exceeds budget."""
        result = self._make_result("c1", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=1500.0)
        self.assertFalse(result.passes_budget(self.balanced_budget))
    
    def test_fails_budget_low_pnl(self):
        """Test that result fails when mean_pnl is below minimum."""
        result = self._make_result("c1", mean_pnl=10.0, kill_prob=0.05, drawdown_cvar=500.0)
        self.assertFalse(result.passes_budget(self.balanced_budget))
    
    def test_fails_budget_evidence_not_verified(self):
        """Test that result fails when evidence is not verified."""
        result = self._make_result(
            "c1", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0,
            evidence_verified=False
        )
        self.assertFalse(result.passes_budget(self.balanced_budget))
    
    def test_boundary_values_pass(self):
        """Test that boundary values exactly meeting budget pass."""
        result = self._make_result(
            "c1",
            mean_pnl=20.0,      # Exactly at min
            kill_prob=0.10,    # Exactly at max
            drawdown_cvar=1000.0,  # Exactly at max
        )
        self.assertTrue(result.passes_budget(self.balanced_budget))
    
    def test_filter_by_budget(self):
        """Test filtering a list of results by budget."""
        results = [
            self._make_result("c1", mean_pnl=50.0, kill_prob=0.05, drawdown_cvar=500.0),  # Pass
            self._make_result("c2", mean_pnl=10.0, kill_prob=0.05, drawdown_cvar=500.0),  # Fail (low pnl)
            self._make_result("c3", mean_pnl=50.0, kill_prob=0.15, drawdown_cvar=500.0),  # Fail (high kill)
            self._make_result("c4", mean_pnl=30.0, kill_prob=0.08, drawdown_cvar=800.0),  # Pass
        ]
        
        filtered = filter_by_budget(results, self.balanced_budget)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].candidate_id, "c1")
        self.assertEqual(filtered[1].candidate_id, "c4")


class TestBudgetConfig(unittest.TestCase):
    """Test BudgetConfig loading."""
    
    def test_load_default_budgets(self):
        """Test loading default budgets."""
        budgets = load_default_budgets()
        
        self.assertIn("balanced", budgets.tiers)
        self.assertIn("conservative", budgets.tiers)
        self.assertIn("aggressive", budgets.tiers)
        
        balanced = budgets.tiers["balanced"]
        self.assertEqual(balanced.max_kill_prob, 0.10)
        self.assertEqual(balanced.max_drawdown_cvar, 1000.0)
        self.assertEqual(balanced.min_mean_pnl, 20.0)
    
    def test_budget_config_from_dict(self):
        """Test BudgetConfig.from_dict()."""
        data = {
            "tiers": {
                "custom": {
                    "max_kill_prob": 0.25,
                    "max_drawdown_cvar": 5000.0,
                    "min_mean_pnl": 100.0,
                }
            }
        }
        
        budgets = BudgetConfig.from_dict(data)
        
        self.assertIn("custom", budgets.tiers)
        custom = budgets.tiers["custom"]
        self.assertEqual(custom.max_kill_prob, 0.25)
        self.assertEqual(custom.max_drawdown_cvar, 5000.0)
        self.assertEqual(custom.min_mean_pnl, 100.0)


if __name__ == "__main__":
    unittest.main()

