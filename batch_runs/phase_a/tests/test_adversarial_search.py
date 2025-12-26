"""
Tests for CEM adversarial search and suite generation.

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
    CEMDistribution,
    generate_cem_candidates,
    generate_smoke_candidates,
    create_fallback_candidate,
    compute_adversarial_score,
    rank_results_deterministic,
    select_elite,
    write_generated_suite_yaml,
    write_candidate_scenarios,
    BASE_PROFILES,
    CEM_PARAMS,
    ADVERSARIAL_PARAM_BOUNDS,
)


class TestCEMDistribution(unittest.TestCase):
    """Test CEM distribution initialization and updates."""
    
    def test_initialize_distribution(self):
        """Test that distribution initializes at center of bounds."""
        dist = CEMDistribution.initialize()
        
        # Check all params have means
        for param in CEM_PARAMS:
            self.assertIn(param, dist.means)
            self.assertIn(param, dist.stds)
            
            lo, hi = ADVERSARIAL_PARAM_BOUNDS[param]
            expected_mean = (lo + hi) / 2.0
            expected_std = (hi - lo) / 4.0
            
            self.assertAlmostEqual(dist.means[param], expected_mean, places=6)
            self.assertAlmostEqual(dist.stds[param], expected_std, places=6)
        
        # Check profile weights sum to 1
        total = sum(dist.profile_weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)
    
    def test_sample_candidate_deterministic(self):
        """Test that sampling is deterministic given seed."""
        import random
        
        dist = CEMDistribution.initialize()
        
        rng1 = random.Random(42)
        cand1 = dist.sample_candidate(rng1, "test1", 100, 200)
        
        rng2 = random.Random(42)
        cand2 = dist.sample_candidate(rng2, "test2", 100, 200)
        
        # Same RNG seed should produce same params
        self.assertEqual(cand1.vol, cand2.vol)
        self.assertEqual(cand1.vol_multiplier, cand2.vol_multiplier)
        self.assertEqual(cand1.profile, cand2.profile)
    
    def test_sample_candidate_bounded(self):
        """Test that sampled candidates respect parameter bounds."""
        import random
        
        dist = CEMDistribution.initialize()
        rng = random.Random(42)
        
        # Sample many candidates
        for i in range(100):
            cand = dist.sample_candidate(rng, f"test_{i}", 1000 + i, 200)
            
            for param in CEM_PARAMS:
                lo, hi = ADVERSARIAL_PARAM_BOUNDS[param]
                value = getattr(cand, param)
                self.assertGreaterEqual(value, lo, f"{param} below min")
                self.assertLessEqual(value, hi, f"{param} above max")
    
    def test_update_from_elite_shifts_mean(self):
        """Test that elite update shifts distribution toward elite."""
        dist = CEMDistribution.initialize()
        original_vol_mean = dist.means["vol"]
        
        # Create elite candidates with high vol
        elite = []
        for i in range(5):
            cand = AdversarialCandidate(
                candidate_id=f"elite_{i}",
                seed=i,
                profile="balanced",
                vol=0.045,  # Near upper bound
                vol_multiplier=2.5,
                jump_intensity=0.001,
                jump_sigma=0.05,
                spread_bps=3.0,
                latency_ms=20.0,
                init_q_tao=0.0,
                daily_loss_limit=1500.0,
                ticks=200,
            )
            elite.append(cand)
        
        dist.update_from_elite(elite, learning_rate=0.5)
        
        # Mean should have shifted toward elite (higher vol)
        self.assertGreater(dist.means["vol"], original_vol_mean)
        # Should be between original and elite mean
        elite_vol_mean = 0.045
        self.assertLess(dist.means["vol"], elite_vol_mean)
    
    def test_update_from_elite_deterministic(self):
        """Test that CEM update is deterministic."""
        elite = []
        for i in range(3):
            cand = AdversarialCandidate(
                candidate_id=f"elite_{i}",
                seed=i,
                profile="aggressive" if i % 2 == 0 else "conservative",
                vol=0.03 + i * 0.005,
                vol_multiplier=2.0,
                jump_intensity=0.001,
                jump_sigma=0.05,
                spread_bps=2.0,
                latency_ms=10.0,
                init_q_tao=float(i * 10),
                daily_loss_limit=1000.0,
                ticks=200,
            )
            elite.append(cand)
        
        dist1 = CEMDistribution.initialize()
        dist1.update_from_elite(elite, learning_rate=0.5)
        
        dist2 = CEMDistribution.initialize()
        dist2.update_from_elite(elite, learning_rate=0.5)
        
        # Should produce identical results
        for param in CEM_PARAMS:
            self.assertEqual(dist1.means[param], dist2.means[param])
            self.assertEqual(dist1.stds[param], dist2.stds[param])
        
        for profile in BASE_PROFILES:
            self.assertEqual(
                dist1.profile_weights[profile],
                dist2.profile_weights[profile]
            )
    
    def test_update_from_empty_elite_noop(self):
        """Test that empty elite list doesn't change distribution."""
        dist = CEMDistribution.initialize()
        original_means = dict(dist.means)
        original_stds = dict(dist.stds)
        
        dist.update_from_elite([], learning_rate=0.5)
        
        for param in CEM_PARAMS:
            self.assertEqual(dist.means[param], original_means[param])
            self.assertEqual(dist.stds[param], original_stds[param])
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        dist = CEMDistribution.initialize()
        dist.means["vol"] = 0.025
        dist.stds["vol"] = 0.008
        
        data = dist.to_dict()
        restored = CEMDistribution.from_dict(data)
        
        self.assertEqual(dist.means["vol"], restored.means["vol"])
        self.assertEqual(dist.stds["vol"], restored.stds["vol"])


class TestCEMCandidateGeneration(unittest.TestCase):
    """Test CEM candidate generation."""
    
    def test_generate_cem_candidates_deterministic(self):
        """Test that CEM candidate generation is deterministic."""
        dist = CEMDistribution.initialize()
        
        cands1 = generate_cem_candidates(dist, 10, 42, iteration=0, ticks=200)
        cands2 = generate_cem_candidates(dist, 10, 42, iteration=0, ticks=200)
        
        self.assertEqual(len(cands1), len(cands2))
        for c1, c2 in zip(cands1, cands2):
            self.assertEqual(c1.seed, c2.seed)
            self.assertEqual(c1.profile, c2.profile)
            self.assertAlmostEqual(c1.vol, c2.vol)
    
    def test_generate_cem_candidates_different_iterations(self):
        """Test that different iterations produce different candidates."""
        dist = CEMDistribution.initialize()
        
        cands_iter0 = generate_cem_candidates(dist, 5, 42, iteration=0, ticks=200)
        cands_iter1 = generate_cem_candidates(dist, 5, 42, iteration=1, ticks=200)
        
        # Seeds should differ between iterations
        seeds0 = [c.seed for c in cands_iter0]
        seeds1 = [c.seed for c in cands_iter1]
        self.assertNotEqual(seeds0, seeds1)
    
    def test_generate_smoke_candidates_fixed(self):
        """Test that smoke candidates are fixed."""
        cands1 = generate_smoke_candidates(42, BASE_PROFILES, 100)
        cands2 = generate_smoke_candidates(42, BASE_PROFILES, 100)
        
        self.assertEqual(len(cands1), 3)  # One per profile
        
        for c1, c2 in zip(cands1, cands2):
            self.assertEqual(c1.seed, c2.seed)
            self.assertEqual(c1.profile, c2.profile)
            self.assertEqual(c1.vol, c2.vol)


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
        self.assertEqual(len(hash1), 8)
    
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
            seed=43,
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


class TestEliteSelection(unittest.TestCase):
    """Test elite selection for CEM."""
    
    def _make_result(self, seed: int, score: float) -> SearchResult:
        """Helper to create test SearchResult with fixed score."""
        candidate = AdversarialCandidate(
            candidate_id=f"cand_{seed}",
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
            candidate_id=f"cand_{seed}",
            candidate=candidate,
            evidence_verified=True,
        )
        result.adversarial_score = score
        return result
    
    def test_select_elite_top_fraction(self):
        """Test that elite selection returns top fraction."""
        results = [self._make_result(i, float(i * 10)) for i in range(20)]
        
        elite = select_elite(results, elite_frac=0.2)
        
        # 20% of 20 = 4 candidates
        self.assertEqual(len(elite), 4)
        
        # Should be the highest scoring ones
        elite_seeds = {c.seed for c in elite}
        expected_seeds = {19, 18, 17, 16}  # Highest scores
        self.assertEqual(elite_seeds, expected_seeds)
    
    def test_select_elite_at_least_one(self):
        """Test that at least one elite is selected."""
        results = [self._make_result(0, 100.0)]
        
        elite = select_elite(results, elite_frac=0.1)
        
        self.assertEqual(len(elite), 1)
    
    def test_select_elite_filters_invalid(self):
        """Test that invalid results are filtered out."""
        results = [self._make_result(i, float(i * 10)) for i in range(10)]
        # Mark some as invalid
        results[9].evidence_verified = False
        results[8].evidence_verified = False
        
        elite = select_elite(results, elite_frac=0.2)
        
        # Should only consider valid results (8 valid, 20% = 1.6 -> 1)
        for cand in elite:
            self.assertIn(cand.seed, range(8))  # Only valid ones


class TestTopKSelection(unittest.TestCase):
    """Test top-K selection with tiebreaks."""
    
    def _make_result(self, seed: int, score: float, candidate_id: str = None) -> SearchResult:
        """Helper to create test SearchResult."""
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
        
        self.assertEqual(ranked[0].candidate.seed, 50)
        self.assertEqual(ranked[1].candidate.seed, 75)
        self.assertEqual(ranked[2].candidate.seed, 100)
    
    def test_tiebreak_by_candidate_id(self):
        """Test final tiebreak by candidate_id."""
        results = [
            self._make_result(42, score=200.0, candidate_id="cand_c"),
            self._make_result(42, score=200.0, candidate_id="cand_a"),
            self._make_result(42, score=200.0, candidate_id="cand_b"),
        ]
        
        ranked = rank_results_deterministic(results)
        
        self.assertEqual(ranked[0].candidate_id, "cand_a")
        self.assertEqual(ranked[1].candidate_id, "cand_b")
        self.assertEqual(ranked[2].candidate_id, "cand_c")


class TestNonEmptySuiteGeneration(unittest.TestCase):
    """Test non-empty suite generation guarantees."""
    
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
            candidates_dir = Path(tmpdir) / "candidates"
            suite_dir = Path(tmpdir) / "suite"
            
            write_candidate_scenarios([r.candidate for r in results], candidates_dir)
            suite_path = write_generated_suite_yaml(results, suite_dir, candidates_dir)
            
            content = suite_path.read_text()
            
            self.assertIn("scenarios:", content)
            self.assertIn("- path:", content)
            self.assertEqual(content.count("- path:"), 3)
    
    def test_suite_yaml_empty_raises(self):
        """Test that empty results raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates_dir = Path(tmpdir) / "candidates"
            suite_dir = Path(tmpdir) / "suite"
            candidates_dir.mkdir(parents=True)
            
            with self.assertRaises(ValueError) as ctx:
                write_generated_suite_yaml([], suite_dir, candidates_dir)
            
            self.assertIn("empty", str(ctx.exception).lower())
    
    def test_fallback_candidate_valid(self):
        """Test that fallback candidate is always valid."""
        fallback = create_fallback_candidate(42, 200)
        
        self.assertEqual(fallback.seed, 42)
        self.assertEqual(fallback.profile, "balanced")
        self.assertEqual(fallback.ticks, 200)
        
        # Should generate valid YAML
        yaml_content = fallback.to_scenario_yaml()
        self.assertIn("scenario_id:", yaml_content)
        self.assertIn("fallback", fallback.candidate_id)


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
    
    def test_suite_yaml_sorted_by_score(self):
        """Test that suite scenarios are sorted by score descending."""
        results = [
            self._make_result(1, score=100.0),
            self._make_result(2, score=300.0),
            self._make_result(3, score=200.0),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates_dir = Path(tmpdir) / "candidates"
            suite_dir = Path(tmpdir) / "suite"
            
            write_candidate_scenarios([r.candidate for r in results], candidates_dir)
            suite_path = write_generated_suite_yaml(results, suite_dir, candidates_dir)
            
            content = suite_path.read_text()
            lines = content.split("\n")
            
            rank_lines = [l for l in lines if "Rank" in l and "score=" in l]
            self.assertEqual(len(rank_lines), 3)
            
            self.assertIn("score=300", rank_lines[0])
            self.assertIn("score=200", rank_lines[1])
            self.assertIn("score=100", rank_lines[2])
    
    def test_suite_yaml_path_based_only(self):
        """Test that suite uses path-based scenarios (no inline env_overrides)."""
        results = [self._make_result(42, score=100.0)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            candidates_dir = Path(tmpdir) / "candidates"
            suite_dir = Path(tmpdir) / "suite"
            
            write_candidate_scenarios([r.candidate for r in results], candidates_dir)
            suite_path = write_generated_suite_yaml(results, suite_dir, candidates_dir)
            
            content = suite_path.read_text()
            
            self.assertIn("- path:", content)
            self.assertNotIn("env_overrides:", content)


class TestScenarioYAMLContent(unittest.TestCase):
    """Test scenario YAML content generation."""
    
    def test_scenario_yaml_content_valid(self):
        """Test that generated scenario YAML has required fields."""
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
        
        self.assertIn("scenario_id:", yaml_content)
        self.assertIn("scenario_version: 1", yaml_content)
        self.assertIn("engine: rl_sim_env", yaml_content)
        self.assertIn("horizon:", yaml_content)
        self.assertIn("steps: 300", yaml_content)
        self.assertIn("risk_profile: conservative", yaml_content)
        self.assertIn("init_q_tao: -20.00", yaml_content)
        self.assertIn("market_model:", yaml_content)
        self.assertIn("jump_diffusion_stub", yaml_content)
        self.assertIn("CEM adversarial search", yaml_content)


if __name__ == "__main__":
    unittest.main()
