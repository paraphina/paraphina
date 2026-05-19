import json
import tempfile
import unittest
from pathlib import Path

from tools import v2_shadow_decision_validator as validator


def valid_row() -> dict:
    return {
        "event_type": "V2_SHADOW_DECISION",
        "schema_version": 1,
        "telemetry_schema_version": 2,
        "now_ms": 1779218759993,
        "decision_mode": "shadow",
        "admission_status": "HOLD",
        "admission_reason": "shadow_only_no_order_authority",
        "can_mutate_orders": False,
        "order_intent_output_count": 0,
        "baseline_plan_intent_count": 1,
        "baseline_mm_order_creating_intent_count": 1,
        "pair_edge_is_admission": False,
        "pressure_complete_claim": False,
        "blocker_cleared": False,
        "require_phase51_gate": True,
        "pair_conditioned_admission_enabled": False,
        "fast_hedge_enabled": False,
        "order_intent_enabled": False,
        "candidates": [
            {
                "candidate_id": "v2_shadow_intent_v1:0:lighter:Buy:0",
                "venue_index": 0,
                "venue_id": "lighter",
                "side": "Buy",
                "price": 250.0,
                "size": 0.01,
                "target_linkage_state": "present_redacted",
                "admission_status": "HOLD",
                "admission_reason": "shadow_only_no_order_authority",
            }
        ],
        "pair_edges": [
            {
                "snapshot_id": "v2_pair_edge_v1:missing_ask",
                "bid_candidate_id": "v2_shadow_intent_v1:0:lighter:Buy:0",
                "ask_candidate_id": None,
                "edge_usd": None,
                "edge_bps": None,
                "feature_only": True,
                "invalid_reason": "missing_ask",
            }
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestV2ShadowDecisionValidator(unittest.TestCase):
    def test_valid_shadow_rows_pass_and_write_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = root / "v2_shadow_decisions.jsonl"
            telemetry = root / "telemetry.jsonl"
            summary = root / "summary.json"
            manifest_path = root / "manifest.json"
            write_jsonl(evidence, [valid_row()])
            telemetry.write_text('{"schema_version": 2}\n', encoding="utf-8")
            summary.write_text('{"ok": true}\n', encoding="utf-8")

            exit_code = validator.main(
                [
                    "--v2-shadow-decisions",
                    str(evidence),
                    "--telemetry",
                    str(telemetry),
                    "--summary",
                    str(summary),
                    "--manifest-output",
                    str(manifest_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_type"], "v2_shadow_decision_evidence_manifest")
            self.assertEqual(manifest["decision_validation_status"], "pass")
            self.assertEqual(manifest["validation"]["row_count"], 1)
            self.assertEqual(manifest["validation"]["candidate_count_total"], 1)
            self.assertFalse(manifest["validation"]["can_mutate_orders_any"])
            self.assertFalse(manifest["validation"]["blocker_cleared_any"])
            self.assertFalse(manifest["validation"]["pressure_complete_claim_any"])
            self.assertEqual(manifest["governance"]["gate_status"], "HOLD")
            self.assertTrue(manifest["governance"]["no_live_flag"])
            self.assertFalse(manifest["governance"]["approved_for_live"])
            self.assertFalse(manifest["governance"]["approved_for_canary"])
            self.assertFalse(manifest["governance"]["live_orders_allowed"])
            self.assertFalse(manifest["governance"]["blocker_cleared"])
            self.assertEqual(len(manifest["files"]), 3)
            for file_info in manifest["files"]:
                self.assertFalse(Path(file_info["path"]).is_absolute())
                self.assertNotIn("..", Path(file_info["path"]).parts)

    def test_rejects_any_order_authority_or_false_clearance(self):
        for field_name, bad_value in [
            ("can_mutate_orders", True),
            ("order_intent_output_count", 1),
            ("admission_status", "PASS"),
            ("pair_edge_is_admission", True),
            ("pressure_complete_claim", True),
            ("blocker_cleared", True),
            ("require_phase51_gate", False),
            ("pair_conditioned_admission_enabled", True),
            ("fast_hedge_enabled", True),
            ("order_intent_enabled", True),
        ]:
            with self.subTest(field_name=field_name):
                row = valid_row()
                row[field_name] = bad_value
                with tempfile.TemporaryDirectory() as td:
                    evidence = Path(td) / "evidence.jsonl"
                    write_jsonl(evidence, [row])
                    with self.assertRaises(validator.ContractViolation):
                        validator.validate_v2_shadow_decisions(evidence)

    def test_rejects_raw_identifier_fields_and_values(self):
        bad_rows = []

        row_with_raw_field = valid_row()
        row_with_raw_field["candidates"][0]["client_order_id"] = "redacted-looking-value"
        bad_rows.append(row_with_raw_field)

        row_with_raw_value = valid_row()
        row_with_raw_value["candidates"][0]["candidate_id"] = (
            "v2_shadow_intent_v1:raw-client-id-must-not-emit"
        )
        bad_rows.append(row_with_raw_value)

        row_with_volume_quota = valid_row()
        row_with_volume_quota["volume_quota_remaining"] = 100
        bad_rows.append(row_with_volume_quota)

        for row in bad_rows:
            with tempfile.TemporaryDirectory() as td:
                evidence = Path(td) / "evidence.jsonl"
                write_jsonl(evidence, [row])
                with self.assertRaises(validator.ContractViolation):
                    validator.validate_v2_shadow_decisions(evidence)

    def test_rejects_candidate_admission_and_unredacted_target_linkage(self):
        row = valid_row()
        row["candidates"][0]["admission_status"] = "PASS"
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td) / "evidence.jsonl"
            write_jsonl(evidence, [row])
            with self.assertRaises(validator.ContractViolation):
                validator.validate_v2_shadow_decisions(evidence)

        row = valid_row()
        row["candidates"][0]["target_linkage_state"] = "canonical_group_id_present"
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td) / "evidence.jsonl"
            write_jsonl(evidence, [row])
            with self.assertRaises(validator.ContractViolation):
                validator.validate_v2_shadow_decisions(evidence)

    def test_rejects_pair_edge_as_admission_authority(self):
        row = valid_row()
        row["pair_edges"][0]["feature_only"] = False
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td) / "evidence.jsonl"
            write_jsonl(evidence, [row])
            with self.assertRaises(validator.ContractViolation):
                validator.validate_v2_shadow_decisions(evidence)

    def test_rejects_pair_edge_references_not_emitted_by_row(self):
        row = valid_row()
        row["pair_edges"][0]["bid_candidate_id"] = "v2_shadow_intent_v1:0:lighter:Sell:99"
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td) / "evidence.jsonl"
            write_jsonl(evidence, [row])
            with self.assertRaises(validator.ContractViolation):
                validator.validate_v2_shadow_decisions(evidence)

    def test_default_requires_at_least_one_candidate_and_mm_creating_intent(self):
        row = valid_row()
        row["candidates"] = []
        row["baseline_mm_order_creating_intent_count"] = 0
        row["pair_edges"] = [
            {
                "snapshot_id": "v2_pair_edge_v1:missing_bid",
                "bid_candidate_id": None,
                "ask_candidate_id": None,
                "edge_usd": None,
                "edge_bps": None,
                "feature_only": True,
                "invalid_reason": "missing_bid",
            }
        ]
        with tempfile.TemporaryDirectory() as td:
            evidence = Path(td) / "evidence.jsonl"
            write_jsonl(evidence, [row])
            with self.assertRaises(validator.ContractViolation):
                validator.validate_v2_shadow_decisions(evidence)

            summary = validator.validate_v2_shadow_decisions(
                evidence,
                require_candidate=False,
                require_mm_creating_intent=False,
            )
            self.assertEqual(summary.row_count, 1)
            self.assertEqual(summary.rows_with_candidates, 0)
            self.assertEqual(summary.rows_with_baseline_mm_order_creating_intents, 0)

    def test_manifest_output_rejects_artifacts_outside_manifest_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = root / "outside"
            inside = root / "inside"
            outside.mkdir()
            inside.mkdir()
            evidence = outside / "v2_shadow_decisions.jsonl"
            write_jsonl(evidence, [valid_row()])

            exit_code = validator.main(
                [
                    "--v2-shadow-decisions",
                    str(evidence),
                    "--manifest-output",
                    str(inside / "manifest.json"),
                ]
            )

            self.assertEqual(exit_code, 2)
            self.assertFalse((inside / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
