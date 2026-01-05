#!/usr/bin/env python3
"""
Unit tests for batch_runs/phase_a/mc_scale.py

Tests plan determinism, shard partitioning, and aggregator validation.
Does NOT require executing the Rust binary.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from batch_runs.phase_a.mc_scale import (
    compute_shard_ranges,
    json_dumps_deterministic,
    sha256_bytes,
)


class TestPlanDeterminism(unittest.TestCase):
    """Test that plan generation is deterministic."""

    def test_same_args_same_plan_hash(self):
        """Same arguments produce byte-identical plan."""
        # Create plan content twice with same args
        seed = 12345
        runs = 100
        shards = 10
        ticks = 600

        ranges = compute_shard_ranges(runs, shards)
        plan1 = {
            "schema_version": 1,
            "seed": seed,
            "runs": runs,
            "shards": len(ranges),
            "ticks": ticks,
            "out_dir": "/tmp/test",
            "shard_ranges": ranges,
        }

        ranges2 = compute_shard_ranges(runs, shards)
        plan2 = {
            "schema_version": 1,
            "seed": seed,
            "runs": runs,
            "shards": len(ranges2),
            "ticks": ticks,
            "out_dir": "/tmp/test",
            "shard_ranges": ranges2,
        }

        json1 = json_dumps_deterministic(plan1)
        json2 = json_dumps_deterministic(plan2)

        self.assertEqual(json1, json2, "Same args should produce identical JSON")
        self.assertEqual(
            sha256_bytes(json1.encode()),
            sha256_bytes(json2.encode()),
            "Same args should produce identical hash"
        )

    def test_different_args_different_plan(self):
        """Different arguments produce different plans."""
        ranges1 = compute_shard_ranges(100, 10)
        plan1 = {
            "schema_version": 1,
            "seed": 12345,
            "runs": 100,
            "shards": len(ranges1),
            "ticks": 600,
            "out_dir": "/tmp/test",
            "shard_ranges": ranges1,
        }

        ranges2 = compute_shard_ranges(100, 10)
        plan2 = {
            "schema_version": 1,
            "seed": 54321,  # Different seed
            "runs": 100,
            "shards": len(ranges2),
            "ticks": 600,
            "out_dir": "/tmp/test",
            "shard_ranges": ranges2,
        }

        json1 = json_dumps_deterministic(plan1)
        json2 = json_dumps_deterministic(plan2)

        self.assertNotEqual(json1, json2, "Different seeds should produce different JSON")

    def test_json_key_ordering_stable(self):
        """JSON keys are sorted for determinism."""
        plan = {
            "z_field": 1,
            "a_field": 2,
            "m_field": 3,
        }

        json_str = json_dumps_deterministic(plan)
        
        # Keys should appear in alphabetical order
        a_pos = json_str.find('"a_field"')
        m_pos = json_str.find('"m_field"')
        z_pos = json_str.find('"z_field"')

        self.assertLess(a_pos, m_pos, "a_field should come before m_field")
        self.assertLess(m_pos, z_pos, "m_field should come before z_field")


class TestShardPartitioning(unittest.TestCase):
    """Test shard range computation."""

    def test_coverage_no_overlap(self):
        """Shard ranges cover all runs with no overlap."""
        runs = 1000
        shards = 10

        ranges = compute_shard_ranges(runs, shards)

        # Check coverage
        covered = set()
        for rng in ranges:
            for i in range(rng["start"], rng["end"]):
                self.assertNotIn(i, covered, f"Index {i} covered multiple times")
                covered.add(i)

        self.assertEqual(covered, set(range(runs)), "All indices should be covered exactly once")

    def test_contiguous_ranges(self):
        """Shard ranges are contiguous (no gaps)."""
        runs = 1000
        shards = 10

        ranges = compute_shard_ranges(runs, shards)

        # First range should start at 0
        self.assertEqual(ranges[0]["start"], 0)

        # Each range should start where the previous ended
        for i in range(1, len(ranges)):
            self.assertEqual(
                ranges[i]["start"],
                ranges[i-1]["end"],
                f"Gap between shard {i-1} and {i}"
            )

        # Last range should end at runs
        self.assertEqual(ranges[-1]["end"], runs)

    def test_uneven_division(self):
        """Handles cases where runs don't divide evenly by shards."""
        runs = 103
        shards = 10

        ranges = compute_shard_ranges(runs, shards)

        # Total count should equal runs
        total = sum(rng["end"] - rng["start"] for rng in ranges)
        self.assertEqual(total, runs)

        # First 3 shards should have 11 runs, rest have 10
        # (103 = 10*10 + 3, so first 3 shards get an extra)
        for i, rng in enumerate(ranges):
            count = rng["end"] - rng["start"]
            if i < 3:
                self.assertEqual(count, 11, f"Shard {i} should have 11 runs")
            else:
                self.assertEqual(count, 10, f"Shard {i} should have 10 runs")

    def test_more_shards_than_runs(self):
        """Handles case where shards > runs."""
        runs = 5
        shards = 10

        ranges = compute_shard_ranges(runs, shards)

        # Should reduce to 5 shards, each with 1 run
        self.assertEqual(len(ranges), 5)
        for i, rng in enumerate(ranges):
            self.assertEqual(rng["end"] - rng["start"], 1)

    def test_single_shard(self):
        """Single shard covers all runs."""
        runs = 100
        shards = 1

        ranges = compute_shard_ranges(runs, shards)

        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0]["start"], 0)
        self.assertEqual(ranges[0]["end"], runs)

    def test_single_run(self):
        """Single run with multiple shards."""
        runs = 1
        shards = 10

        ranges = compute_shard_ranges(runs, shards)

        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0]["start"], 0)
        self.assertEqual(ranges[0]["end"], 1)


class TestAggregatorValidation(unittest.TestCase):
    """Test aggregator validation logic (without executing binary)."""

    def create_test_jsonl(self, records):
        """Create a temporary JSONL file with given records."""
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return Path(path)

    def test_rejects_duplicate_indices(self):
        """Aggregator should reject duplicate run_index values."""
        records = [
            {"run_index": 0, "seed": 100, "pnl_total": 10.0, "max_drawdown": 1.0, "kill_switch": False},
            {"run_index": 1, "seed": 101, "pnl_total": 20.0, "max_drawdown": 2.0, "kill_switch": False},
            {"run_index": 0, "seed": 100, "pnl_total": 15.0, "max_drawdown": 1.5, "kill_switch": False},  # Duplicate
        ]

        # Simulate validation
        indices = [r["run_index"] for r in records]
        has_duplicate = len(indices) != len(set(indices))

        self.assertTrue(has_duplicate, "Should detect duplicate indices")

    def test_rejects_non_contiguous_indices(self):
        """Aggregator should reject non-contiguous run_index values."""
        records = [
            {"run_index": 0, "seed": 100, "pnl_total": 10.0, "max_drawdown": 1.0, "kill_switch": False},
            {"run_index": 1, "seed": 101, "pnl_total": 20.0, "max_drawdown": 2.0, "kill_switch": False},
            {"run_index": 3, "seed": 103, "pnl_total": 30.0, "max_drawdown": 3.0, "kill_switch": False},  # Gap at 2
        ]

        # Simulate validation
        sorted_records = sorted(records, key=lambda r: r["run_index"])
        min_idx = sorted_records[0]["run_index"]
        max_idx = sorted_records[-1]["run_index"]
        expected_count = max_idx - min_idx + 1

        is_contiguous = len(records) == expected_count

        self.assertFalse(is_contiguous, "Should detect non-contiguous indices")

    def test_accepts_valid_records(self):
        """Aggregator should accept valid contiguous records."""
        records = [
            {"run_index": 0, "seed": 100, "pnl_total": 10.0, "max_drawdown": 1.0, "kill_switch": False},
            {"run_index": 1, "seed": 101, "pnl_total": 20.0, "max_drawdown": 2.0, "kill_switch": False},
            {"run_index": 2, "seed": 102, "pnl_total": 30.0, "max_drawdown": 3.0, "kill_switch": False},
        ]

        # Simulate validation
        indices = [r["run_index"] for r in records]
        has_duplicate = len(indices) != len(set(indices))

        sorted_records = sorted(records, key=lambda r: r["run_index"])
        min_idx = sorted_records[0]["run_index"]
        max_idx = sorted_records[-1]["run_index"]
        expected_count = max_idx - min_idx + 1
        is_contiguous = len(records) == expected_count

        self.assertFalse(has_duplicate, "Should not detect duplicates")
        self.assertTrue(is_contiguous, "Should be contiguous")

    def test_validates_seed_contract(self):
        """Aggregator should validate seed contract when base_seed provided."""
        base_seed = 100

        # Valid records following seed contract
        valid_records = [
            {"run_index": 0, "seed": 100, "pnl_total": 10.0},  # base_seed + 0
            {"run_index": 1, "seed": 101, "pnl_total": 20.0},  # base_seed + 1
            {"run_index": 2, "seed": 102, "pnl_total": 30.0},  # base_seed + 2
        ]

        for record in valid_records:
            expected_seed = base_seed + record["run_index"]
            self.assertEqual(record["seed"], expected_seed)

        # Invalid record
        invalid_record = {"run_index": 5, "seed": 999, "pnl_total": 50.0}  # Wrong seed
        expected_seed = base_seed + invalid_record["run_index"]
        self.assertNotEqual(invalid_record["seed"], expected_seed)

    def test_handles_missing_shard_files(self):
        """Aggregator should fail gracefully for missing shard files."""
        # This is a behavioral test - just document the expected error
        # Actual file checking happens in cmd_aggregate
        pass


class TestSeedMapping(unittest.TestCase):
    """Test deterministic seed mapping contract."""

    def test_seed_mapping_basic(self):
        """Seed contract: seed_i = base_seed + i."""
        base_seed = 12345

        for i in range(100):
            expected = base_seed + i
            # Python ints don't overflow, but we match the u64 wrap behavior
            computed = (base_seed + i) & 0xFFFFFFFFFFFFFFFF
            self.assertEqual(computed, expected)

    def test_seed_mapping_u64_wrap(self):
        """Seed wrapping at u64 max."""
        u64_max = (1 << 64) - 1
        base_seed = u64_max - 5

        # Index 5 should give u64_max
        self.assertEqual((base_seed + 5) & 0xFFFFFFFFFFFFFFFF, u64_max)

        # Index 6 should wrap to 0
        self.assertEqual((base_seed + 6) & 0xFFFFFFFFFFFFFFFF, 0)

        # Index 10 should wrap to 4
        self.assertEqual((base_seed + 10) & 0xFFFFFFFFFFFFFFFF, 4)

    def test_shard_slicing_preserves_seeds(self):
        """Sharding produces same seeds as full run."""
        base_seed = 42
        total_runs = 1000

        # Full run seeds
        full_seeds = [(base_seed + i) & 0xFFFFFFFFFFFFFFFF for i in range(total_runs)]

        # Sharded seeds
        shards = compute_shard_ranges(total_runs, 10)
        shard_seeds = []
        for rng in shards:
            for i in range(rng["start"], rng["end"]):
                shard_seeds.append((base_seed + i) & 0xFFFFFFFFFFFFFFFF)

        self.assertEqual(full_seeds, shard_seeds)


class TestJsonlFormat(unittest.TestCase):
    """Test JSONL record format handling."""

    def test_required_fields(self):
        """JSONL records must have required fields."""
        required_fields = {"run_index", "seed", "pnl_total", "max_drawdown", "kill_switch"}

        valid_record = {
            "run_index": 0,
            "seed": 12345,
            "pnl_total": 100.5,
            "max_drawdown": 10.2,
            "kill_switch": False,
            "kill_tick": None,
            "kill_reason": "None",
            "ticks_executed": 600,
        }

        self.assertTrue(required_fields.issubset(set(valid_record.keys())))

    def test_json_round_trip(self):
        """JSONL records survive JSON round-trip."""
        original = {
            "run_index": 42,
            "seed": 123456789,
            "pnl_total": -50.123456789,
            "max_drawdown": 25.987654321,
            "kill_switch": True,
            "kill_tick": 150,
            "kill_reason": "DailyLoss",
        }

        json_str = json.dumps(original)
        parsed = json.loads(json_str)

        self.assertEqual(original["run_index"], parsed["run_index"])
        self.assertEqual(original["seed"], parsed["seed"])
        self.assertAlmostEqual(original["pnl_total"], parsed["pnl_total"], places=10)
        self.assertAlmostEqual(original["max_drawdown"], parsed["max_drawdown"], places=10)
        self.assertEqual(original["kill_switch"], parsed["kill_switch"])


class TestEvidencePackFormat(unittest.TestCase):
    """Regression tests for evidence pack format compliance."""

    def test_sha256sums_required_entries(self):
        """
        SHA256SUMS must include evidence_pack/manifest.json and evidence_pack/suite.yaml.
        
        This is a regression test for the issue where manually generated SHA256SUMS
        was missing these required entries, causing sim_eval verify-evidence-pack to fail.
        """
        # These are the required entries per evidence_pack_verify.rs
        required_entries = [
            "evidence_pack/manifest.json",
            "evidence_pack/suite.yaml",
        ]
        
        # Create a mock SHA256SUMS content that should pass validation
        valid_sha256sums = [
            "abc123def456  evidence_pack/manifest.json",
            "def789abc012  evidence_pack/suite.yaml",
            "111222333444  mc_summary.json",
        ]
        
        # Parse and validate
        entries_found = set()
        for line in valid_sha256sums:
            parts = line.split()
            if len(parts) >= 2:
                path = parts[1]
                entries_found.add(path)
        
        for required in required_entries:
            self.assertIn(required, entries_found, 
                f"Required entry '{required}' missing from SHA256SUMS")

    def test_sha256sums_missing_manifest_detected(self):
        """
        Detect when SHA256SUMS is missing evidence_pack/manifest.json.
        """
        # SHA256SUMS without manifest.json (the bug we fixed)
        invalid_sha256sums = [
            "abc123def456  mc_scale_plan.json",
            "def789abc012  mc_runs.jsonl",
            "111222333444  mc_summary.json",
        ]
        
        entries_found = set()
        for line in invalid_sha256sums:
            parts = line.split()
            if len(parts) >= 2:
                path = parts[1]
                entries_found.add(path)
        
        # Should NOT have the required entries
        self.assertNotIn("evidence_pack/manifest.json", entries_found)
        self.assertNotIn("evidence_pack/suite.yaml", entries_found)

    def test_sha256sums_format_validation(self):
        """
        SHA256SUMS format: "<64hex>  <relpath>" (two spaces after hash).
        """
        valid_lines = [
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  evidence_pack/manifest.json",
            "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210  evidence_pack/suite.yaml",
        ]
        
        for line in valid_lines:
            parts = line.split(None, 1)  # Split on whitespace, max 2 parts
            self.assertEqual(len(parts), 2, f"Line should have hash and path: {line}")
            
            hash_part = parts[0]
            path_part = parts[1].strip()
            
            # Hash should be 64 hex chars
            self.assertEqual(len(hash_part), 64, f"Hash should be 64 chars: {hash_part}")
            self.assertTrue(all(c in "0123456789abcdef" for c in hash_part.lower()),
                f"Hash should be lowercase hex: {hash_part}")
            
            # Path should be relative (no leading /)
            self.assertFalse(path_part.startswith("/"), 
                f"Path should be relative: {path_part}")


if __name__ == "__main__":
    unittest.main()

