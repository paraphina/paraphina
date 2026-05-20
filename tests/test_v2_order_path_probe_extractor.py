import json
import tempfile
import unittest
from pathlib import Path

from tools import v2_order_path_probe_extractor as extractor


def gate_state():
    return {
        "enabled": True,
        "decision_mode_is_paper_admission": False,
        "decision_mode_is_live_canary_admission": True,
        "execution_mode_is_paper": False,
        "execution_mode_is_live": True,
        "pair_edge_enabled": True,
        "pair_conditioned_admission_enabled": True,
        "order_intent_enabled": True,
        "fast_hedge_disabled": True,
        "require_phase51_gate": True,
        "live_canary_admission_approved": True,
        "live_canary_order_path_probe_approved": True,
        "live_canary_mode_enabled": True,
        "live_canary_profile_metadata_present": True,
        "live_canary_max_position_present": True,
        "live_canary_max_gross_position_present": True,
        "live_canary_max_abs_venue_position_present": True,
        "live_canary_max_open_orders_present": True,
        "live_canary_post_only_enforced": True,
        "live_canary_reduce_only_not_enforced": True,
    }


def probe_row():
    return {
        "event_type": "V2_ADMISSION_DECISION",
        "schema_version": 1,
        "telemetry_schema_version": 2,
        "now_ms": 1779226270000,
        "decision_mode": "live_canary_admission",
        "execution_mode": "live",
        "authority_scope": "live_canary_single_venue_order_path_probe",
        "admission_status": "ADMITTED",
        "admission_reason": "live_canary_single_venue_order_path_probe",
        "can_filter_existing_intents": True,
        "can_create_new_intents": False,
        "can_mutate_live_orders": False,
        "order_intent_output_count": 1,
        "baseline_plan_intent_count": 1,
        "baseline_mm_order_creating_intent_count": 1,
        "suppressed_mm_order_creating_intent_count": 0,
        "pair_edge_is_admission": False,
        "order_path_probe_is_admission": True,
        "pressure_complete_claim": False,
        "blocker_cleared": False,
        "gate_state": gate_state(),
        "ranking_schema_version": 1,
        "ranking_feature_only": False,
        "ranking_is_admission": False,
        "pair_edges": [
            {
                "snapshot_id": "v2_pair_edge_v1:missing_ask",
                "bid_candidate_id": "v2_shadow_intent_v1:0:lighter:buy:0",
                "ask_candidate_id": None,
                "edge_usd": None,
                "edge_bps": None,
                "feature_only": False,
                "invalid_reason": "missing_ask",
            }
        ],
        "admitted_candidates": [
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
        ],
    }


def hold_row():
    row = probe_row()
    row["authority_scope"] = "live_canary_ranked_admission"
    row["admission_status"] = "HOLD"
    row["admission_reason"] = "no_positive_ranked_candidates"
    row["order_intent_output_count"] = 0
    row["baseline_plan_intent_count"] = 0
    row["baseline_mm_order_creating_intent_count"] = 0
    row["pair_edge_is_admission"] = False
    row["order_path_probe_is_admission"] = False
    row["admitted_candidates"] = []
    row["gate_state"]["live_canary_order_path_probe_approved"] = False
    return row


def write_rows(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class TestV2OrderPathProbeExtractor(unittest.TestCase):
    def test_extracts_single_probe_with_source_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v2_decisions.jsonl"
            output = root / "probe.jsonl"
            manifest_path = root / "manifest.json"
            write_rows(source, [hold_row(), probe_row(), hold_row()])

            manifest = extractor.extract_probe(source, output, manifest_path, run_root=root)

            self.assertTrue(output.exists())
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["extraction"]["source_line_no"], 2)
            self.assertEqual(manifest["validation"]["row_count"], 1)
            self.assertEqual(manifest["validation"]["live_canary_order_path_probe_rows"], 1)
            self.assertEqual(manifest["governance"]["gate_status"], "LIVE_CANARY_ORDER_PATH_PROBE")
            self.assertFalse(manifest["governance"]["approved_for_promotion"])
            self.assertFalse(manifest["governance"]["blocker_cleared"])
            extracted = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(extracted), 1)
            self.assertTrue(extracted[0]["order_path_probe_is_admission"])

    def test_rejects_missing_probe_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v2_decisions.jsonl"
            write_rows(source, [hold_row(), hold_row()])
            with self.assertRaises(extractor.V2OrderPathProbeExtractionError):
                extractor.extract_probe(source, root / "probe.jsonl", root / "manifest.json")

    def test_rejects_multiple_probe_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v2_decisions.jsonl"
            write_rows(source, [probe_row(), probe_row()])
            with self.assertRaises(extractor.V2OrderPathProbeExtractionError):
                extractor.extract_probe(source, root / "probe.jsonl", root / "manifest.json")

    def test_rejects_false_clearance_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "v2_decisions.jsonl"
            row = probe_row()
            row["blocker_cleared"] = True
            write_rows(source, [row])
            output = root / "probe.jsonl"
            with self.assertRaises(Exception):
                extractor.extract_probe(source, output, root / "manifest.json")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
