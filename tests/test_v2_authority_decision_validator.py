import json
import tempfile
import unittest
from pathlib import Path

from tools import v2_authority_decision_validator as validator


def paper_gate_state():
    return {
        "enabled": True,
        "decision_mode_is_paper_admission": True,
        "decision_mode_is_live_canary_admission": False,
        "execution_mode_is_paper": True,
        "execution_mode_is_live": False,
        "pair_edge_enabled": True,
        "pair_conditioned_admission_enabled": True,
        "order_intent_enabled": True,
        "fast_hedge_disabled": True,
        "require_phase51_gate": True,
        "live_canary_admission_approved": False,
        "live_canary_order_path_probe_approved": False,
        "live_canary_mode_enabled": False,
        "live_canary_profile_metadata_present": False,
        "live_canary_max_position_present": False,
        "live_canary_max_gross_position_present": False,
        "live_canary_max_abs_venue_position_present": False,
        "live_canary_max_open_orders_present": False,
        "live_canary_post_only_enforced": False,
        "live_canary_reduce_only_not_enforced": False,
        "live_canary_baseline_hedge_authority_acknowledged": False,
    }


def live_canary_gate_state():
    gate = paper_gate_state()
    gate.update(
        {
            "decision_mode_is_paper_admission": False,
            "decision_mode_is_live_canary_admission": True,
            "execution_mode_is_paper": False,
            "execution_mode_is_live": True,
            "live_canary_admission_approved": True,
            "live_canary_mode_enabled": True,
            "live_canary_profile_metadata_present": True,
            "live_canary_max_position_present": True,
            "live_canary_max_gross_position_present": True,
            "live_canary_max_abs_venue_position_present": True,
            "live_canary_max_open_orders_present": True,
            "live_canary_post_only_enforced": True,
            "live_canary_reduce_only_not_enforced": True,
            "live_canary_baseline_hedge_authority_acknowledged": True,
        }
    )
    return gate


def admitted_row():
    return {
        "event_type": "V2_ADMISSION_DECISION",
        "schema_version": 1,
        "telemetry_schema_version": 2,
        "now_ms": 1779226270000,
        "decision_mode": "paper_admission",
        "execution_mode": "paper",
        "authority_scope": "paper_only",
        "admission_status": "ADMITTED",
        "admission_reason": "paper_positive_pair_edge_ranked_admission",
        "can_filter_existing_intents": True,
        "can_create_new_intents": False,
        "can_mutate_live_orders": False,
        "order_intent_output_count": 1,
        "baseline_plan_intent_count": 2,
        "baseline_mm_order_creating_intent_count": 2,
        "suppressed_mm_order_creating_intent_count": 1,
        "pair_edge_is_admission": True,
        "order_path_probe_is_admission": False,
        "pressure_complete_claim": False,
        "blocker_cleared": False,
        "gate_state": paper_gate_state(),
        "ranking_schema_version": 1,
        "ranking_feature_only": False,
        "ranking_is_admission": True,
        "pair_edges": [
            {
                "snapshot_id": "v2_pair_edge_v1:v2_shadow_intent_v1:0:extended:buy:0:v2_shadow_intent_v1:1:lighter:sell:1",
                "bid_candidate_id": "v2_shadow_intent_v1:0:extended:buy:0",
                "ask_candidate_id": "v2_shadow_intent_v1:1:lighter:sell:1",
                "edge_usd": 1.0,
                "edge_bps": 2.0,
                "feature_only": False,
                "invalid_reason": None,
            }
        ],
        "admitted_candidates": [
            {
                "candidate_id": "v2_shadow_intent_v1:0:extended:buy:0",
                "venue_index": 0,
                "venue_id": "extended",
                "side": "Buy",
                "rank_index": 1,
                "rank_score_microusd": 1_000_000,
                "pair_edge_feature_usd": 1.0,
                "pair_edge_feature_bps": 2.0,
                "reference_candidate_id": "v2_shadow_intent_v1:2:aster:buy:2",
            }
        ],
    }


def live_canary_admitted_row():
    row = admitted_row()
    row["decision_mode"] = "live_canary_admission"
    row["execution_mode"] = "live"
    row["authority_scope"] = "live_canary_ranked_admission"
    row["admission_reason"] = "live_canary_positive_pair_edge_ranked_admission"
    row["gate_state"] = live_canary_gate_state()
    return row


def live_canary_order_path_probe_row():
    row = live_canary_admitted_row()
    row["authority_scope"] = "live_canary_single_venue_order_path_probe"
    row["admission_reason"] = "live_canary_single_venue_order_path_probe"
    row["baseline_plan_intent_count"] = 1
    row["baseline_mm_order_creating_intent_count"] = 1
    row["suppressed_mm_order_creating_intent_count"] = 0
    row["pair_edge_is_admission"] = False
    row["order_path_probe_is_admission"] = True
    row["ranking_is_admission"] = False
    row["gate_state"]["live_canary_order_path_probe_approved"] = True
    row["pair_edges"] = [
        {
            "snapshot_id": "v2_pair_edge_v1:missing_ask",
            "bid_candidate_id": "v2_shadow_intent_v1:0:lighter:buy:0",
            "ask_candidate_id": None,
            "edge_usd": None,
            "edge_bps": None,
            "feature_only": False,
            "invalid_reason": "missing_ask",
        }
    ]
    row["admitted_candidates"] = [
        {
            "candidate_id": "v2_shadow_intent_v1:0:lighter:buy:0",
            "venue_index": 0,
            "venue_id": "lighter",
            "side": "Buy",
            "rank_index": 1,
            "rank_score_microusd": 0,
            "pair_edge_feature_usd": None,
            "pair_edge_feature_bps": None,
            "reference_candidate_id": None,
        }
    ]
    return row


def write_rows(root: Path, rows):
    path = root / "v2_authority_decisions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class TestV2AuthorityDecisionValidator(unittest.TestCase):
    def test_accepts_paper_only_admission_row_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = write_rows(root, [admitted_row()])
            manifest = root / "manifest.json"

            summary = validator.validate_v2_authority_decisions(evidence)
            validator.write_manifest(evidence, manifest, summary)

            self.assertEqual(summary.row_count, 1)
            self.assertEqual(summary.admitted_rows, 1)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["governance"]["gate_status"], "PAPER_ONLY")
            self.assertFalse(data["governance"]["approved_for_live"])
            self.assertFalse(data["governance"]["blocker_cleared"])

    def test_accepts_live_canary_ranked_admission_row_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = write_rows(root, [live_canary_admitted_row()])
            manifest = root / "manifest.json"

            summary = validator.validate_v2_authority_decisions(evidence)
            validator.write_manifest(evidence, manifest, summary)

            self.assertEqual(summary.row_count, 1)
            self.assertEqual(summary.admitted_rows, 1)
            self.assertEqual(summary.live_canary_rows, 1)
            self.assertFalse(summary.can_mutate_live_orders_any)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["governance"]["gate_status"], "LIVE_CANARY")
            self.assertTrue(data["governance"]["approved_for_canary"])
            self.assertTrue(data["governance"]["live_orders_allowed"])
            self.assertFalse(data["governance"]["approved_for_live"])
            self.assertFalse(data["governance"]["blocker_cleared"])
            self.assertEqual(
                data["v2_authority_contract"]["authority_scope"],
                "live_canary_ranked_admission",
            )
            self.assertFalse(data["v2_authority_contract"]["can_create_new_intents"])
            self.assertFalse(data["v2_authority_contract"]["fast_hedge_enabled"])

    def test_accepts_live_canary_order_path_probe_as_non_promotion_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = write_rows(root, [live_canary_order_path_probe_row()])
            manifest = root / "manifest.json"

            summary = validator.validate_v2_authority_decisions(evidence)
            validator.write_manifest(evidence, manifest, summary)

            self.assertEqual(summary.row_count, 1)
            self.assertEqual(summary.admitted_rows, 1)
            self.assertEqual(summary.live_canary_order_path_probe_rows, 1)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["governance"]["gate_status"], "LIVE_CANARY_ORDER_PATH_PROBE")
            self.assertTrue(data["governance"]["probe_only"])
            self.assertFalse(data["governance"]["approved_for_promotion"])
            self.assertFalse(data["governance"]["approved_for_live"])
            self.assertFalse(data["governance"]["capital_change_allowed"])
            self.assertFalse(data["governance"]["blocker_cleared"])
            self.assertEqual(
                data["v2_authority_contract"]["authority_scope"],
                "live_canary_single_venue_order_path_probe",
            )
            self.assertTrue(data["v2_authority_contract"]["order_path_probe_only"])

    def test_rejects_multiple_order_path_probe_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = write_rows(
                Path(tmp),
                [
                    live_canary_order_path_probe_row(),
                    live_canary_order_path_probe_row(),
                ],
            )
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_rejects_mixed_order_path_probe_and_ranked_admission_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = write_rows(
                Path(tmp),
                [
                    live_canary_order_path_probe_row(),
                    live_canary_admitted_row(),
                ],
            )
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_rejects_mixed_order_path_probe_and_ranked_hold_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            hold = live_canary_admitted_row()
            hold["admission_status"] = "HOLD"
            hold["admission_reason"] = "no_positive_ranked_candidates"
            hold["order_intent_output_count"] = 0
            hold["suppressed_mm_order_creating_intent_count"] = 1
            hold["pair_edge_is_admission"] = False
            hold["ranking_is_admission"] = False
            hold["admitted_candidates"] = []
            evidence = write_rows(Path(tmp), [live_canary_order_path_probe_row(), hold])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_rejects_order_path_probe_if_mislabeled_as_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = live_canary_order_path_probe_row()
            row["ranking_is_admission"] = True
            evidence = write_rows(Path(tmp), [row])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_rejects_synthesized_or_false_clearance_authority(self):
        for field in ["can_create_new_intents", "blocker_cleared", "pressure_complete_claim"]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                row = admitted_row()
                row[field] = True
                evidence = write_rows(Path(tmp), [row])
                with self.assertRaises(validator.V2AuthorityValidationError):
                    validator.validate_v2_authority_decisions(evidence)

    def test_rejects_paper_row_with_live_mutation_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = admitted_row()
            row["can_mutate_live_orders"] = True
            evidence = write_rows(Path(tmp), [row])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_rejects_admitted_row_with_missing_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = admitted_row()
            row["gate_state"]["order_intent_enabled"] = False
            row["can_filter_existing_intents"] = False
            evidence = write_rows(Path(tmp), [row])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_rejects_live_canary_without_baseline_hedge_authority_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = live_canary_admitted_row()
            row["gate_state"]["live_canary_baseline_hedge_authority_acknowledged"] = False
            evidence = write_rows(Path(tmp), [row])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_accepts_hold_row_with_missing_gate_without_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = admitted_row()
            row["admission_status"] = "HOLD"
            row["admission_reason"] = "paper_admission_gate_not_satisfied"
            row["order_intent_output_count"] = 0
            row["suppressed_mm_order_creating_intent_count"] = 0
            row["pair_edge_is_admission"] = False
            row["ranking_is_admission"] = False
            row["can_filter_existing_intents"] = False
            row["gate_state"]["order_intent_enabled"] = False
            row["admitted_candidates"] = []
            evidence = write_rows(Path(tmp), [row])
            summary = validator.validate_v2_authority_decisions(evidence)
            self.assertEqual(summary.hold_rows, 1)

    def test_rejects_raw_identifier_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = admitted_row()
            row["admitted_candidates"][0]["client_order_id"] = "raw-client"
            evidence = write_rows(Path(tmp), [row])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_rejects_v2_authority_in_non_paper_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = admitted_row()
            row["execution_mode"] = "live"
            evidence = write_rows(Path(tmp), [row])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_rejects_live_canary_row_with_missing_profile_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = live_canary_admitted_row()
            row["gate_state"]["live_canary_profile_metadata_present"] = False
            row["can_filter_existing_intents"] = False
            evidence = write_rows(Path(tmp), [row])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_accepts_live_canary_hold_without_order_path_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = live_canary_admitted_row()
            row["admission_status"] = "HOLD"
            row["admission_reason"] = "no_positive_ranked_candidates"
            row["order_intent_output_count"] = 0
            row["suppressed_mm_order_creating_intent_count"] = row[
                "baseline_mm_order_creating_intent_count"
            ]
            row["pair_edge_is_admission"] = False
            row["ranking_is_admission"] = False
            row["admitted_candidates"] = []
            evidence = write_rows(Path(tmp), [row])
            summary = validator.validate_v2_authority_decisions(evidence)
            self.assertEqual(summary.live_canary_rows, 1)
            self.assertEqual(summary.live_canary_order_path_probe_rows, 0)

    def test_rejects_count_mismatch(self):
        for field, value in [
            ("order_intent_output_count", 2),
            ("suppressed_mm_order_creating_intent_count", 0),
        ]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                row = admitted_row()
                row[field] = value
                evidence = write_rows(Path(tmp), [row])
                with self.assertRaises(validator.V2AuthorityValidationError):
                    validator.validate_v2_authority_decisions(evidence)

    def test_rejects_duplicate_admitted_candidate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = admitted_row()
            row["admitted_candidates"].append(dict(row["admitted_candidates"][0]))
            row["order_intent_output_count"] = 2
            row["baseline_mm_order_creating_intent_count"] = 3
            row["suppressed_mm_order_creating_intent_count"] = 1
            evidence = write_rows(Path(tmp), [row])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

    def test_accepts_ranked_admission_without_positive_pair_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = admitted_row()
            row["pair_edges"][0]["edge_usd"] = 0.0
            row["pair_edge_is_admission"] = False
            evidence = write_rows(Path(tmp), [row])
            summary = validator.validate_v2_authority_decisions(evidence)
            self.assertEqual(summary.admitted_rows, 1)

    def test_rejects_manifest_artifact_outside_manifest_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            manifest_root = root / "manifest_root"
            outside.mkdir()
            manifest_root.mkdir()
            evidence = write_rows(outside, [admitted_row()])
            summary = validator.validate_v2_authority_decisions(evidence)
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.write_manifest(evidence, manifest_root / "manifest.json", summary)


if __name__ == "__main__":
    unittest.main()
