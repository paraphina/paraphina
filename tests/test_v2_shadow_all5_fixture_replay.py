import json
import tempfile
import unittest
from pathlib import Path

from tools import v2_shadow_all5_fixture_replay as replay
from tools import v2_shadow_decision_validator as validator
from tools import v2_shadow_scenario_matrix as matrix


class TestV2ShadowAllFiveFixtureReplay(unittest.TestCase):
    def test_build_fixture_replay_uses_all_five_sanitized_top_of_book_inputs(self):
        rows = replay.build_fixture_replay(Path.cwd())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["scenario_id"], "all_five_fixture_replay_top_of_book")
        self.assertEqual(row["admission_status"], "HOLD")
        self.assertFalse(row["can_mutate_orders"])
        self.assertFalse(row["blocker_cleared"])
        self.assertFalse(row["pressure_complete_claim"])
        self.assertEqual(row["baseline_mm_order_creating_intent_count"], 10)
        self.assertEqual(len(row["candidates"]), 10)
        self.assertEqual({candidate["venue_id"] for candidate in row["candidates"]}, set(matrix.VENUES))
        self.assertEqual(
            {candidate["target_linkage_state"] for candidate in row["candidates"]},
            {"missing"},
        )
        self.assertEqual(
            {candidate["candidate_source"] for candidate in row["candidates"]},
            {"mm_quote"},
        )
        self.assertEqual(
            {candidate["price_size_source"] for candidate in row["candidates"]},
            {"quote_level"},
        )
        self.assertEqual(len(row["candidate_rankings"]), 10)
        self.assertEqual(len(row["pair_edges"]), 1)
        self.assertTrue(row["pair_edges"][0]["feature_only"])

    def test_generate_fixture_replay_writes_valid_hold_only_manifest_pack(self):
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "replay"
            paths = replay.generate_fixture_replay(output_root, repo_root=Path.cwd())

            self.assertTrue(paths["decision_path"].exists())
            self.assertTrue(paths["summary_path"].exists())
            self.assertTrue(paths["manifest_path"].exists())

            validation = validator.validate_v2_shadow_decisions(
                paths["decision_path"],
                require_ev_evaluations=True,
            )
            self.assertEqual(validation.row_count, 11)
            self.assertEqual(validation.shadow_decision_row_count, 1)
            self.assertEqual(validation.ev_evaluation_count_total, 10)
            self.assertEqual(validation.candidate_count_total, 10)
            self.assertEqual(validation.candidate_ranking_count_total, 10)
            self.assertEqual(validation.pair_edge_count_total, 1)
            self.assertFalse(validation.blocker_cleared_any)
            self.assertFalse(validation.pressure_complete_claim_any)

            summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
            self.assertEqual(summary["artifact_type"], "v2_shadow_all5_fixture_replay_summary")
            self.assertEqual(summary["venues"], list(matrix.VENUES))
            self.assertEqual(summary["target_linkage_state"], "missing")
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertFalse(summary["blocker_cleared"])
            self.assertFalse(summary["live_orders_allowed"])
            self.assertTrue(summary["fixtures_are_sanitized_local_inputs"])

            manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["decision_validation_status"], "pass")
            self.assertEqual(manifest["validation"]["ev_evaluation_count_total"], 10)
            self.assertEqual(manifest["governance"]["gate_status"], "HOLD")
            self.assertFalse(manifest["governance"]["approved_for_live"])

    def test_cli_writes_fixture_replay(self):
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "replay"
            exit_code = replay.main(["--output-root", str(output_root), "--repo-root", str(Path.cwd())])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_root / "v2_shadow_fixture_replay.jsonl").exists())
            self.assertTrue((output_root / "manifest.json").exists())

    def test_fixture_replay_tool_has_no_runtime_or_credential_surface(self):
        source = Path(replay.__file__).read_text(encoding="utf-8")
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
