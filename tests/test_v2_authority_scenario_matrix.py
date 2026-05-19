import tempfile
import unittest
from pathlib import Path

from tools import v2_authority_decision_validator as validator
from tools import v2_authority_scenario_matrix as matrix


class TestV2AuthorityScenarioMatrix(unittest.TestCase):
    def test_matrix_contains_valid_admission_and_hold_cases(self):
        rows = matrix.build_matrix()
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(rows[0]["admission_status"], "ADMITTED")
        self.assertTrue(rows[0]["pair_edge_is_admission"])
        self.assertEqual(
            {candidate["venue_id"] for candidate in rows[0]["admitted_candidates"]},
            {"extended", "hyperliquid", "aster", "lighter", "paradex"},
        )
        hold_reasons = {row["admission_reason"] for row in rows[1:]}
        self.assertIn("paper_admission_gate_not_satisfied", hold_reasons)
        self.assertIn("no_positive_pair_edge", hold_reasons)
        self.assertIn("missing_bid", hold_reasons)
        self.assertIn("missing_ask", hold_reasons)
        self.assertIn("no_admitted_candidates", hold_reasons)

    def test_matrix_output_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(matrix.main(["--output-root", str(root)]), 0)
            evidence = root / "v2_authority_decisions.jsonl"
            manifest = root / "manifest.json"
            summary = validator.validate_v2_authority_decisions(evidence)
            validator.write_manifest(evidence, manifest, summary)
            self.assertTrue(manifest.exists())
            self.assertEqual(summary.admitted_rows, 1)
            self.assertGreater(summary.hold_rows, 0)


if __name__ == "__main__":
    unittest.main()
