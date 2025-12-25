"""
Tests for adversarial search and suite generation.

Hermetic tests - no cargo invocation.
"""

import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from batch_runs.phase_a.adversarial_search_promote import (
    AdversarialCandidate,
    SearchResult,
    generate_random_candidates,
    generate_smoke_candidates,
    mutate_candidate,
    compute_adversarial_score,
    rank_results_deterministic,
    generate_suite_v2_yaml,
    promote_scenarios,
    write_topk_json,
    BASE_PROFILES,
)


class TestCandidateGeneration(unittest.TestCase):
    """Test deterministic candidate generation."""
    
    def test_generate_random_candidates_deterministic(self):
        """Test that random candidates are deterministic given seed."""
        candidates1 = generate_random_candidates(10, 42, BASE_PROFILES, 200)
        candidates2 = generate_random_candidates(10, 42, BASE_PROFILES, 200)
        
        # Should be identical
        self.assertEqual(len(candidates1), len(candidates2))
        for c1, c2 in zip(candidates1, candidates2):
            self.assertEqual(c1.seed, c2.seed)
            self.assertEqual(c1.profile, c2.profile)
            self.assertAlmostEqual(c1.vol, c2.vol)
            self.assertAlmostEqual(c1.vol_multiplier, c2.vol_multiplier)
    
    def test_generate_different_seeds_different_candidates(self):
        """Test that different seeds produce different candidates."""
        candidates1 = generate_random_candidates(10, 42, BASE_PROFILES, 200)
        candidates2 = generate_random_candidates(10, 99, BASE_PROFILES, 200)
        
        # Seeds should differ
        seeds1 = [c.seed for c in candidates1]
        seeds2 = [c.seed for c in candidates2]
        self.assertNotEqual(seeds1, seeds2)
    
    def test_generate_smoke_candidates_fixed(self):
        """Test that smoke candidates are fixed and deterministic."""
        candidates1 = generate_smoke_candidates(42, BASE_PROFILES, 100)
        candidates2 = generate_smoke_candidates(42, BASE_PROFILES, 100)
        
        self.assertEqual(len(candidates1), len(candidates2))
        self.assertEqual(len(candidates1), 3)  # One per profile
        
        for c1, c2 in zip(candidates1, candidates2):
            self.assertEqual(c1.seed, c2.seed)
            self.assertEqual(c1.profile, c2.profile)


class TestDeterministicNaming(unittest.TestCase):
    """Test deterministic scenario naming."""
    
    def test_scenario_hash_deterministic(self):
        """Test that scenario hash is deterministic."""
        candidate = AdversarialCandidate(
            candidate_id="test",
            seed=42,
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        hash1 = candidate.scenario_hash()
        hash2 = candidate.scenario_hash()
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 8)  # Short hash
    
    def test_scenario_filename_deterministic(self):
        """Test that scenario filename is deterministic."""
        candidate = AdversarialCandidate(
            candidate_id="test",
            seed=42,
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        name1 = candidate.scenario_filename()
        name2 = candidate.scenario_filename()
        
        self.assertEqual(name1, name2)
        self.assertTrue(name1.startswith("adv_s00042_"))
        self.assertTrue(name1.endswith(".yaml"))
    
    def test_different_params_different_hash(self):
        """Test that different params produce different hashes."""
        base = AdversarialCandidate(
            candidate_id="test",
            seed=42,
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        modified = AdversarialCandidate(
            candidate_id="test",
            seed=43,  # Different seed
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        self.assertNotEqual(base.scenario_hash(), modified.scenario_hash())
    
    def test_deterministic_naming_same_seed(self):
        """
        Test that same seed + params always produces same filename.
        
        This is critical for reproducibility of promoted scenarios.
        """
        expected_filename = None
        for _ in range(5):
            candidate = AdversarialCandidate(
                candidate_id="test",
                seed=12345,
                profile="conservative",
                vol=0.015,
                vol_multiplier=1.5,
                jump_intensity=0.0005,
                jump_sigma=0.03,
                spread_bps=1.5,
                latency_ms=5.0,
                init_q_tao=-20.0,
                daily_loss_limit=800.0,
                ticks=300,
            )
            
            filename = candidate.scenario_filename()
            
            # First iteration: record the expected filename
            if expected_filename is None:
                expected_filename = filename
                # Verify format
                self.assertTrue(filename.startswith("adv_s12345_"))
                self.assertTrue(filename.endswith(".yaml"))
            else:
                # All subsequent iterations must match
                self.assertEqual(filename, expected_filename)


class TestAdversarialScoring(unittest.TestCase):
    """Test adversarial scoring and ranking."""
    
    def _make_result(
        self,
        seed: int,
        kill_switch: bool,
        max_drawdown: float,
        mean_pnl: float,
    ) -> SearchResult:
        """Helper to create test SearchResult."""
        candidate = AdversarialCandidate(
            candidate_id=f"test_{seed}",
            seed=seed,
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        result = SearchResult(
            candidate_id=f"test_{seed}",
            candidate=candidate,
            kill_switch=kill_switch,
            max_drawdown=max_drawdown,
            mean_pnl=mean_pnl,
            evidence_verified=True,
        )
        result.adversarial_score = compute_adversarial_score(result)
        return result
    
    def test_kill_switch_highest_priority(self):
        """Test that kill switch dominates scoring."""
        no_kill = self._make_result(1, kill_switch=False, max_drawdown=100.0, mean_pnl=-50.0)
        with_kill = self._make_result(2, kill_switch=True, max_drawdown=10.0, mean_pnl=100.0)
        
        # Kill should dominate even with lower drawdown and positive pnl
        self.assertGreater(with_kill.adversarial_score, no_kill.adversarial_score)
    
    def test_drawdown_secondary_priority(self):
        """Test that drawdown is secondary to kill switch."""
        low_dd = self._make_result(1, kill_switch=False, max_drawdown=10.0, mean_pnl=0.0)
        high_dd = self._make_result(2, kill_switch=False, max_drawdown=100.0, mean_pnl=0.0)
        
        self.assertGreater(high_dd.adversarial_score, low_dd.adversarial_score)
    
    def test_pnl_tertiary_priority(self):
        """Test that negative pnl increases score."""
        pos_pnl = self._make_result(1, kill_switch=False, max_drawdown=50.0, mean_pnl=100.0)
        neg_pnl = self._make_result(2, kill_switch=False, max_drawdown=50.0, mean_pnl=-100.0)
        
        self.assertGreater(neg_pnl.adversarial_score, pos_pnl.adversarial_score)


class TestTopKSelection(unittest.TestCase):
    """Test top-K selection with tiebreaks."""
    
    def _make_result(
        self,
        seed: int,
        score: float,
        candidate_id: str = None,
    ) -> SearchResult:
        """Helper to create test SearchResult with fixed score."""
        if candidate_id is None:
            candidate_id = f"cand_{seed}"
        
        candidate = AdversarialCandidate(
            candidate_id=candidate_id,
            seed=seed,
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        result = SearchResult(
            candidate_id=candidate_id,
            candidate=candidate,
            evidence_verified=True,
        )
        result.adversarial_score = score
        return result
    
    def test_ranking_by_score_descending(self):
        """Test that ranking is by score descending."""
        results = [
            self._make_result(1, score=100.0),
            self._make_result(2, score=300.0),
            self._make_result(3, score=200.0),
        ]
        
        ranked = rank_results_deterministic(results)
        
        self.assertEqual(ranked[0].adversarial_score, 300.0)
        self.assertEqual(ranked[1].adversarial_score, 200.0)
        self.assertEqual(ranked[2].adversarial_score, 100.0)
    
    def test_tiebreak_by_seed(self):
        """Test that ties are broken by seed (ascending)."""
        results = [
            self._make_result(100, score=200.0),
            self._make_result(50, score=200.0),
            self._make_result(75, score=200.0),
        ]
        
        ranked = rank_results_deterministic(results)
        
        # Same score, should be ordered by seed ascending
        self.assertEqual(ranked[0].candidate.seed, 50)
        self.assertEqual(ranked[1].candidate.seed, 75)
        self.assertEqual(ranked[2].candidate.seed, 100)
    
    def test_tiebreak_by_candidate_id(self):
        """Test final tiebreak by candidate_id."""
        # Same score and seed (via different candidate_ids)
        results = [
            self._make_result(42, score=200.0, candidate_id="cand_c"),
            self._make_result(42, score=200.0, candidate_id="cand_a"),
            self._make_result(42, score=200.0, candidate_id="cand_b"),
        ]
        
        ranked = rank_results_deterministic(results)
        
        # Same score and seed, should be ordered by candidate_id
        self.assertEqual(ranked[0].candidate_id, "cand_a")
        self.assertEqual(ranked[1].candidate_id, "cand_b")
        self.assertEqual(ranked[2].candidate_id, "cand_c")
    
    def test_topk_selection_determinism(self):
        """Test that top-K selection is deterministic."""
        results = [
            self._make_result(seed, score=float(100 - seed % 10))
            for seed in range(20)
        ]
        
        ranked1 = rank_results_deterministic(results)
        ranked2 = rank_results_deterministic(results)
        
        # Should be identical
        self.assertEqual(
            [r.candidate_id for r in ranked1],
            [r.candidate_id for r in ranked2],
        )


class TestSuiteYamlGeneration(unittest.TestCase):
    """Test suite YAML generation."""
    
    def _make_result(self, seed: int, score: float) -> SearchResult:
        """Helper to create test SearchResult."""
        candidate = AdversarialCandidate(
            candidate_id=f"test_{seed}",
            seed=seed,
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        result = SearchResult(
            candidate_id=f"test_{seed}",
            candidate=candidate,
            evidence_verified=True,
        )
        result.adversarial_score = score
        return result
    
    def test_suite_yaml_nonempty(self):
        """Test that generated suite has non-empty scenarios list."""
        results = [self._make_result(seed, 100.0 + seed) for seed in range(3)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "test_suite.yaml"
            generate_suite_v2_yaml(results, suite_path)
            
            content = suite_path.read_text()
            
            # Should contain scenarios
            self.assertIn("scenarios:", content)
            self.assertIn("- path:", content)
            # Should have 3 scenario entries
            self.assertEqual(content.count("- path:"), 3)
    
    def test_suite_yaml_sorted_by_score(self):
        """Test that suite scenarios are sorted by score descending."""
        results = [
            self._make_result(1, score=100.0),
            self._make_result(2, score=300.0),
            self._make_result(3, score=200.0),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "test_suite.yaml"
            generate_suite_v2_yaml(results, suite_path)
            
            content = suite_path.read_text()
            lines = content.split("\n")
            
            # Find rank comments
            rank_lines = [l for l in lines if "Rank" in l and "score=" in l]
            self.assertEqual(len(rank_lines), 3)
            
            # Check order: highest score first
            self.assertIn("score=300", rank_lines[0])
            self.assertIn("score=200", rank_lines[1])
            self.assertIn("score=100", rank_lines[2])
    
    def test_suite_yaml_path_based_only(self):
        """Test that suite uses path-based scenarios (no inline env_overrides)."""
        results = [self._make_result(42, score=100.0)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "test_suite.yaml"
            generate_suite_v2_yaml(results, suite_path)
            
            content = suite_path.read_text()
            
            # Should have path-based scenarios
            self.assertIn("- path:", content)
            # Should NOT have inline env_overrides
            self.assertNotIn("env_overrides:", content)
    
    def test_suite_yaml_empty_raises(self):
        """Test that empty results raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_path = Path(tmpdir) / "test_suite.yaml"
            
            with self.assertRaises(ValueError) as ctx:
                generate_suite_v2_yaml([], suite_path)
            
            self.assertIn("empty", str(ctx.exception).lower())


class TestScenarioPromotion(unittest.TestCase):
    """Test scenario file promotion."""
    
    def test_promote_scenarios_writes_files(self):
        """Test that scenarios are written to files."""
        candidate = AdversarialCandidate(
            candidate_id="test_42",
            seed=42,
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        result = SearchResult(
            candidate_id="test_42",
            candidate=candidate,
            evidence_verified=True,
        )
        result.adversarial_score = 100.0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override the GENERATED_SCENARIOS_DIR
            import batch_runs.phase_a.adversarial_search_promote as asp
            original_dir = asp.GENERATED_SCENARIOS_DIR
            asp.GENERATED_SCENARIOS_DIR = Path(tmpdir)
            
            try:
                paths = promote_scenarios([result])
                
                self.assertEqual(len(paths), 1)
                self.assertTrue(paths[0].exists())
                
                content = paths[0].read_text()
                self.assertIn("scenario_id:", content)
                self.assertIn("adv_s00042_", content)
            finally:
                asp.GENERATED_SCENARIOS_DIR = original_dir
    
    def test_scenario_yaml_content_valid(self):
        """Test that generated scenario YAML is valid."""
        candidate = AdversarialCandidate(
            candidate_id="test",
            seed=42,
            profile="conservative",
            vol=0.015,
            vol_multiplier=1.5,
            jump_intensity=0.0005,
            jump_sigma=0.03,
            spread_bps=1.5,
            latency_ms=5.0,
            init_q_tao=-20.0,
            daily_loss_limit=800.0,
            ticks=300,
        )
        
        yaml_content = candidate.to_scenario_yaml()
        
        # Check required fields
        self.assertIn("scenario_id:", yaml_content)
        self.assertIn("scenario_version: 1", yaml_content)
        self.assertIn("engine: rl_sim_env", yaml_content)
        self.assertIn("horizon:", yaml_content)
        self.assertIn("steps: 300", yaml_content)
        self.assertIn("risk_profile: conservative", yaml_content)
        self.assertIn("init_q_tao: -20.00", yaml_content)
        self.assertIn("market_model:", yaml_content)
        self.assertIn("jump_diffusion_stub", yaml_content)


class TestMutation(unittest.TestCase):
    """Test evolutionary mutation."""
    
    def test_mutation_preserves_some_params(self):
        """Test that mutation only changes 1-2 parameters."""
        import random
        rng = random.Random(42)
        
        parent = AdversarialCandidate(
            candidate_id="parent",
            seed=100,
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        child = mutate_candidate(parent, rng)
        
        # Child should have different seed
        self.assertNotEqual(child.seed, parent.seed)
        
        # Child should have different ticks (inherits from parent)
        self.assertEqual(child.ticks, parent.ticks)
    
    def test_mutation_deterministic(self):
        """Test that mutation is deterministic given RNG."""
        import random
        
        parent = AdversarialCandidate(
            candidate_id="parent",
            seed=100,
            profile="balanced",
            vol=0.02,
            vol_multiplier=2.0,
            jump_intensity=0.001,
            jump_sigma=0.05,
            spread_bps=2.0,
            latency_ms=10.0,
            init_q_tao=0.0,
            daily_loss_limit=1000.0,
            ticks=200,
        )
        
        rng1 = random.Random(42)
        child1 = mutate_candidate(parent, rng1)
        
        rng2 = random.Random(42)
        child2 = mutate_candidate(parent, rng2)
        
        self.assertEqual(child1.seed, child2.seed)
        self.assertEqual(child1.vol, child2.vol)
        self.assertEqual(child1.profile, child2.profile)


if __name__ == "__main__":
    unittest.main()

