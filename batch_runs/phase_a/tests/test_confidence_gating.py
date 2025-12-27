"""
Tests for confidence-aware budget gating.

Hermetic tests - no cargo invocation.
These tests verify that:
1. Budget gating uses LCB/UCB instead of point estimates
2. A candidate with passing mean but failing bound is rejected
3. Selection is deterministic
"""

import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.promote_pipeline import (
    TierBudget,
    BudgetConfig,
    TrialResult,
    CandidateConfig,
    select_winner,
    load_default_budgets,
    _create_tier_budget,
)


class TestConfidenceAwareGating(unittest.TestCase):
    """Test that gating uses confidence bounds, not point estimates."""
    
    def setUp(self):
        """Create test fixtures."""
        self.balanced_budget = TierBudget(
            tier_name="balanced",
            max_kill_ucb=0.10,
            max_dd_cvar=1000.0,
            min_pnl_lcb_usd=20.0,
            max_kill_prob=0.10,
            max_drawdown_cvar=1000.0,
            min_mean_pnl=20.0,
        )
        
        self.conservative_budget = TierBudget(
            tier_name="conservative",
            max_kill_ucb=0.05,
            max_dd_cvar=500.0,
            min_pnl_lcb_usd=10.0,
            max_kill_prob=0.05,
            max_drawdown_cvar=500.0,
            min_mean_pnl=10.0,
        )
    
    def _make_config(self, candidate_id: str) -> CandidateConfig:
        """Helper to create a test config."""
        return CandidateConfig(
            candidate_id=candidate_id,
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
    
    def _make_result(
        self,
        candidate_id: str,
        pnl_mean: float,
        pnl_stdev: float,
        pnl_lcb: float,
        kill_rate: float,
        kill_ucb: float,
        dd_cvar: float,
        kill_k: int = 0,
        kill_n: int = 50,
        evidence_verified: bool = True,
        research_passed: bool = True,
        adversarial_passed: bool = True,
    ) -> TrialResult:
        """Helper to create a test TrialResult with confidence bounds."""
        return TrialResult(
            trial_id=f"trial_{candidate_id}",
            candidate_id=candidate_id,
            config=self._make_config(candidate_id),
            trial_dir=Path("/tmp/test"),
            mc_mean_pnl=pnl_mean,
            mc_pnl_stdev=pnl_stdev,
            mc_pnl_cvar=-100.0,
            mc_drawdown_cvar=dd_cvar,
            mc_kill_prob_point=kill_rate,
            mc_kill_prob_ci_upper=kill_ucb,
            mc_total_runs=kill_n,
            mc_kill_count=kill_k,
            mc_pnl_lcb=pnl_lcb,
            mc_kill_ucb=kill_ucb,
            evidence_verified=evidence_verified,
            research_passed=research_passed,
            adversarial_passed=adversarial_passed,
        )
    
    def test_passes_with_both_mean_and_bounds_passing(self):
        """Test candidate passes when both mean and bounds pass budget."""
        # pnl_mean=50, pnl_lcb=40 > min_pnl_lcb_usd=20 ✓
        # kill_rate=0.05, kill_ucb=0.08 <= max_kill_ucb=0.10 ✓
        # dd_cvar=500 <= max_dd_cvar=1000 ✓
        result = self._make_result(
            "c1",
            pnl_mean=50.0,
            pnl_stdev=20.0,
            pnl_lcb=40.0,
            kill_rate=0.05,
            kill_ucb=0.08,
            dd_cvar=500.0,
        )
        self.assertTrue(result.passes_budget(self.balanced_budget))
    
    def test_fails_when_mean_passes_but_lcb_fails(self):
        """Test candidate fails when mean passes but LCB fails.
        
        THIS IS THE KEY TEST:
        A candidate with high mean PnL but high uncertainty (wide confidence interval)
        should fail because the lower bound doesn't meet the threshold.
        """
        # pnl_mean=50 > min_mean_pnl=20 ✓ (mean passes)
        # pnl_lcb=15 < min_pnl_lcb_usd=20 ✗ (bound fails!)
        # This candidate MUST fail despite having a good mean
        result = self._make_result(
            "c1",
            pnl_mean=50.0,      # Good mean
            pnl_stdev=100.0,    # High variance
            pnl_lcb=15.0,       # LCB below threshold!
            kill_rate=0.05,
            kill_ucb=0.08,
            dd_cvar=500.0,
        )
        self.assertFalse(result.passes_budget(self.balanced_budget))
    
    def test_fails_when_kill_rate_passes_but_ucb_fails(self):
        """Test candidate fails when kill rate passes but UCB fails.
        
        A candidate with low observed kill rate but high uncertainty
        should fail because the upper bound exceeds the threshold.
        """
        # kill_rate=0.05 <= max_kill_prob=0.10 ✓ (point estimate passes)
        # kill_ucb=0.15 > max_kill_ucb=0.10 ✗ (bound fails!)
        result = self._make_result(
            "c1",
            pnl_mean=50.0,
            pnl_stdev=20.0,
            pnl_lcb=40.0,
            kill_rate=0.05,     # Low observed rate
            kill_ucb=0.15,      # But high UCB due to small sample!
            dd_cvar=500.0,
            kill_k=1,
            kill_n=20,          # Small sample = wide CI
        )
        self.assertFalse(result.passes_budget(self.balanced_budget))
    
    def test_fails_when_dd_cvar_exceeds_budget(self):
        """Test candidate fails when dd_cvar exceeds budget."""
        # dd_cvar=1500 > max_dd_cvar=1000 ✗
        result = self._make_result(
            "c1",
            pnl_mean=50.0,
            pnl_stdev=20.0,
            pnl_lcb=40.0,
            kill_rate=0.05,
            kill_ucb=0.08,
            dd_cvar=1500.0,  # Exceeds budget
        )
        self.assertFalse(result.passes_budget(self.balanced_budget))
    
    def test_fails_with_nan_pnl_lcb(self):
        """Test candidate fails when pnl_lcb is NaN (fail closed)."""
        result = self._make_result(
            "c1",
            pnl_mean=50.0,
            pnl_stdev=20.0,
            pnl_lcb=float("nan"),  # Missing bound!
            kill_rate=0.05,
            kill_ucb=0.08,
            dd_cvar=500.0,
        )
        # is_valid should be False, so passes_budget should also be False
        self.assertFalse(result.is_valid)
        self.assertFalse(result.passes_budget(self.balanced_budget))
    
    def test_fails_with_nan_kill_ucb(self):
        """Test candidate fails when kill_ucb is NaN (fail closed)."""
        result = self._make_result(
            "c1",
            pnl_mean=50.0,
            pnl_stdev=20.0,
            pnl_lcb=40.0,
            kill_rate=0.05,
            kill_ucb=float("nan"),  # Missing bound!
            dd_cvar=500.0,
        )
        self.assertFalse(result.is_valid)
        self.assertFalse(result.passes_budget(self.balanced_budget))
    
    def test_fails_with_missing_evidence_verification(self):
        """Test candidate fails when evidence not verified."""
        result = self._make_result(
            "c1",
            pnl_mean=50.0,
            pnl_stdev=20.0,
            pnl_lcb=40.0,
            kill_rate=0.05,
            kill_ucb=0.08,
            dd_cvar=500.0,
            evidence_verified=False,  # Not verified!
        )
        self.assertFalse(result.is_valid)
        self.assertFalse(result.passes_budget(self.balanced_budget))


class TestDeterministicSelection(unittest.TestCase):
    """Test that selection is deterministic."""
    
    def setUp(self):
        """Create test fixtures."""
        self.budget = TierBudget(
            tier_name="balanced",
            max_kill_ucb=0.10,
            max_dd_cvar=1000.0,
            min_pnl_lcb_usd=20.0,
            max_kill_prob=0.10,
            max_drawdown_cvar=1000.0,
            min_mean_pnl=20.0,
        )
    
    def _make_config(self, candidate_id: str) -> CandidateConfig:
        return CandidateConfig(
            candidate_id=candidate_id,
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
    
    def _make_result(
        self,
        candidate_id: str,
        pnl_mean: float,
        pnl_lcb: float,
        kill_ucb: float,
        dd_cvar: float,
    ) -> TrialResult:
        return TrialResult(
            trial_id=f"trial_{candidate_id}",
            candidate_id=candidate_id,
            config=self._make_config(candidate_id),
            trial_dir=Path("/tmp/test"),
            mc_mean_pnl=pnl_mean,
            mc_pnl_stdev=10.0,
            mc_pnl_cvar=-100.0,
            mc_drawdown_cvar=dd_cvar,
            mc_kill_prob_point=kill_ucb * 0.8,
            mc_kill_prob_ci_upper=kill_ucb,
            mc_total_runs=50,
            mc_kill_count=int(kill_ucb * 50 * 0.8),
            mc_pnl_lcb=pnl_lcb,
            mc_kill_ucb=kill_ucb,
            evidence_verified=True,
            research_passed=True,
            adversarial_passed=True,
        )
    
    def test_same_inputs_same_winner(self):
        """Test that same inputs produce same winner across multiple runs."""
        results = [
            self._make_result("c1", 50.0, 40.0, 0.08, 500.0),
            self._make_result("c2", 60.0, 45.0, 0.07, 600.0),  # Higher pnl
            self._make_result("c3", 55.0, 42.0, 0.06, 400.0),
        ]
        
        winners = []
        for _ in range(10):
            winner = select_winner(results, self.budget)
            winners.append(winner.candidate_id if winner else None)
        
        # All winners should be the same
        self.assertEqual(len(set(winners)), 1)
        self.assertEqual(winners[0], "c2")  # Highest pnl
    
    def test_tie_break_by_candidate_id(self):
        """Test tie-breaking by candidate_id for determinism."""
        # All have same metrics
        results = [
            self._make_result("c_charlie", 50.0, 40.0, 0.08, 500.0),
            self._make_result("c_alpha", 50.0, 40.0, 0.08, 500.0),
            self._make_result("c_bravo", 50.0, 40.0, 0.08, 500.0),
        ]
        
        winner = select_winner(results, self.budget)
        self.assertIsNotNone(winner)
        self.assertEqual(winner.candidate_id, "c_alpha")  # Alphabetically first


class TestBudgetConfigBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility of budget configuration."""
    
    def test_create_tier_budget_from_old_fields(self):
        """Test that old fields are mapped to new fields."""
        tier_data = {
            "max_kill_prob": 0.08,
            "max_drawdown_cvar": 800.0,
            "min_mean_pnl": 15.0,
        }
        
        budget = _create_tier_budget("test", tier_data)
        
        # New fields should match old fields
        self.assertEqual(budget.max_kill_ucb, 0.08)
        self.assertEqual(budget.max_dd_cvar, 800.0)
        self.assertEqual(budget.min_pnl_lcb_usd, 15.0)
    
    def test_create_tier_budget_new_fields_take_precedence(self):
        """Test that new fields take precedence over old fields."""
        tier_data = {
            # Old fields
            "max_kill_prob": 0.08,
            "max_drawdown_cvar": 800.0,
            "min_mean_pnl": 15.0,
            # New fields (should take precedence)
            "max_kill_ucb": 0.10,
            "max_dd_cvar": 1000.0,
            "min_pnl_lcb_usd": 20.0,
        }
        
        budget = _create_tier_budget("test", tier_data)
        
        # New fields should be used
        self.assertEqual(budget.max_kill_ucb, 0.10)
        self.assertEqual(budget.max_dd_cvar, 1000.0)
        self.assertEqual(budget.min_pnl_lcb_usd, 20.0)
    
    def test_default_budgets_have_confidence_fields(self):
        """Test that default budgets include confidence-aware fields."""
        budgets = load_default_budgets()
        
        self.assertEqual(budgets.alpha, 0.05)
        
        for tier_name, tier in budgets.tiers.items():
            self.assertIsNotNone(tier.max_kill_ucb)
            self.assertIsNotNone(tier.max_dd_cvar)
            self.assertIsNotNone(tier.min_pnl_lcb_usd)


class TestIsValidWithConfidenceBounds(unittest.TestCase):
    """Test is_valid property with confidence bound requirements."""
    
    def _make_config(self, candidate_id: str) -> CandidateConfig:
        return CandidateConfig(
            candidate_id=candidate_id,
            profile="balanced",
            hedge_band_base=0.05,
            mm_size_eta=1.0,
            vol_ref=0.10,
            daily_loss_limit=1000.0,
        )
    
    def test_valid_with_all_bounds(self):
        """Test valid when all bounds are present."""
        result = TrialResult(
            trial_id="test",
            candidate_id="test",
            config=self._make_config("test"),
            trial_dir=Path("/tmp/test"),
            mc_mean_pnl=50.0,
            mc_pnl_stdev=10.0,
            mc_pnl_cvar=-100.0,
            mc_drawdown_cvar=500.0,
            mc_kill_prob_point=0.05,
            mc_kill_prob_ci_upper=0.08,
            mc_total_runs=50,
            mc_kill_count=2,
            mc_pnl_lcb=40.0,  # Valid bound
            mc_kill_ucb=0.08,  # Valid bound
            evidence_verified=True,
            research_passed=True,
            adversarial_passed=True,
        )
        self.assertTrue(result.is_valid)
    
    def test_invalid_with_missing_pnl_lcb(self):
        """Test invalid when pnl_lcb is NaN."""
        result = TrialResult(
            trial_id="test",
            candidate_id="test",
            config=self._make_config("test"),
            trial_dir=Path("/tmp/test"),
            mc_mean_pnl=50.0,
            mc_pnl_stdev=10.0,
            mc_pnl_cvar=-100.0,
            mc_drawdown_cvar=500.0,
            mc_kill_prob_point=0.05,
            mc_kill_prob_ci_upper=0.08,
            mc_total_runs=50,
            mc_kill_count=2,
            mc_pnl_lcb=float("nan"),  # Missing!
            mc_kill_ucb=0.08,
            evidence_verified=True,
            research_passed=True,
            adversarial_passed=True,
        )
        self.assertFalse(result.is_valid)
    
    def test_invalid_with_missing_kill_ucb(self):
        """Test invalid when kill_ucb is NaN."""
        result = TrialResult(
            trial_id="test",
            candidate_id="test",
            config=self._make_config("test"),
            trial_dir=Path("/tmp/test"),
            mc_mean_pnl=50.0,
            mc_pnl_stdev=10.0,
            mc_pnl_cvar=-100.0,
            mc_drawdown_cvar=500.0,
            mc_kill_prob_point=0.05,
            mc_kill_prob_ci_upper=0.08,
            mc_total_runs=50,
            mc_kill_count=2,
            mc_pnl_lcb=40.0,
            mc_kill_ucb=float("nan"),  # Missing!
            evidence_verified=True,
            research_passed=True,
            adversarial_passed=True,
        )
        self.assertFalse(result.is_valid)


if __name__ == "__main__":
    unittest.main()

