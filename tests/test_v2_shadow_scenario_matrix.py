import json
import tempfile
import unittest
from pathlib import Path

from tools import v2_shadow_decision_validator as validator
from tools import v2_shadow_scenario_matrix as matrix


class TestV2ShadowScenarioMatrix(unittest.TestCase):
    def test_build_scenarios_cover_all_five_venues_and_pair_edge_states(self):
        rows = matrix.build_scenarios()
        self.assertEqual(len(rows), 4)
        scenario_ids = {row["scenario_id"] for row in rows}
        self.assertEqual(
            scenario_ids,
            {
                "all_five_crossed_pair_edge_feature",
                "all_five_bid_only_missing_ask",
                "all_five_ask_only_missing_bid",
                "intent_fallback_place_replace_candidates",
            },
        )

        venues = set()
        invalid_reasons = set()
        for row in rows:
            self.assertEqual(row["admission_status"], "HOLD")
            self.assertFalse(row["can_mutate_orders"])
            self.assertEqual(row["order_intent_output_count"], 0)
            self.assertFalse(row["blocker_cleared"])
            self.assertFalse(row["pressure_complete_claim"])
            venues.update(candidate["venue_id"] for candidate in row["candidates"])
            invalid_reasons.add(row["pair_edges"][0]["invalid_reason"])

        self.assertEqual(venues, set(matrix.VENUES))
        self.assertEqual(invalid_reasons, {None, "missing_bid", "missing_ask"})

        crossed = next(row for row in rows if row["scenario_id"] == "all_five_crossed_pair_edge_feature")
        pair_edge = crossed["pair_edges"][0]
        self.assertEqual(pair_edge["bid_candidate_id"], "v2_shadow_v1:1:hyperliquid:buy")
        self.assertEqual(pair_edge["ask_candidate_id"], "v2_shadow_v1:2:aster:sell")
        self.assertTrue(pair_edge["feature_only"])
        self.assertGreater(pair_edge["edge_usd"], 0)

        intent_fallback = next(
            row for row in rows if row["scenario_id"] == "intent_fallback_place_replace_candidates"
        )
        self.assertTrue(
            all(
                candidate["candidate_id"].startswith("v2_shadow_intent_v1:")
                for candidate in intent_fallback["candidates"]
            )
        )

    def test_generate_matrix_writes_valid_manifest_pack(self):
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "matrix"
            paths = matrix.generate_matrix(output_root)

            self.assertTrue(paths["decision_path"].exists())
            self.assertTrue(paths["summary_path"].exists())
            self.assertTrue(paths["manifest_path"].exists())

            validation = validator.validate_v2_shadow_decisions(paths["decision_path"])
            self.assertEqual(validation.row_count, 4)
            self.assertEqual(validation.candidate_count_total, 22)
            self.assertEqual(validation.pair_edge_count_total, 4)

            summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
            self.assertEqual(summary["venues"], list(matrix.VENUES))
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertFalse(summary["blocker_cleared"])
            self.assertFalse(summary["live_orders_allowed"])

            manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["decision_validation_status"], "pass")
            self.assertEqual(manifest["governance"]["gate_status"], "HOLD")
            self.assertFalse(manifest["governance"]["approved_for_live"])
            self.assertEqual(
                {file_info["path"] for file_info in manifest["files"]},
                {"v2_shadow_decisions.jsonl", "scenario_summary.json"},
            )

    def test_cli_writes_matrix(self):
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "matrix"
            exit_code = matrix.main(["--output-root", str(output_root)])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "v2_shadow_decisions.jsonl").exists())
            self.assertTrue((output_root / "manifest.json").exists())

    def test_matrix_tool_has_no_runtime_or_credential_surface(self):
        source = Path(matrix.__file__).read_text(encoding="utf-8")
        forbidden_terms = [
            "paraphina_live",
            "systemctl",
            "sendTx",
            "sendTxBatch",
            "place_order",
            "cancel_order",
            "replace_order",
            "LIGHTER_API_PRIVATE_KEY_HEX",
            "/etc/paraphina",
            "subprocess",
        ]
        for term in forbidden_terms:
            with self.subTest(term=term):
                self.assertNotIn(term, source)


if __name__ == "__main__":
    unittest.main()
