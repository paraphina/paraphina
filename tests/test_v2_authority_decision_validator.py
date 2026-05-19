import json
import tempfile
import unittest
from pathlib import Path

from tools import v2_authority_decision_validator as validator


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
        "pressure_complete_claim": False,
        "blocker_cleared": False,
        "gate_state": {
            "enabled": True,
            "decision_mode_is_paper_admission": True,
            "execution_mode_is_paper": True,
            "pair_edge_enabled": True,
            "pair_conditioned_admission_enabled": True,
            "order_intent_enabled": True,
            "fast_hedge_disabled": True,
            "require_phase51_gate": True,
        },
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

    def test_rejects_live_or_capital_authority(self):
        for field in ["can_create_new_intents", "can_mutate_live_orders", "blocker_cleared", "pressure_complete_claim"]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                row = admitted_row()
                row[field] = True
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

    def test_rejects_admitted_row_without_positive_pair_edge(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = admitted_row()
            row["pair_edges"][0]["edge_usd"] = 0.0
            evidence = write_rows(Path(tmp), [row])
            with self.assertRaises(validator.V2AuthorityValidationError):
                validator.validate_v2_authority_decisions(evidence)

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
