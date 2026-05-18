import json
import tempfile
import unittest
from pathlib import Path

from tools import phase51h_observed_pfill_feature_audit as phase51h
from tools import phase51i_pfill_feature_matrix_admissibility as phase51i


BASELINE_COMMIT = phase51h.BASELINE_COMMIT
SOURCE_SHA = "sha256:source-owner-pfill-pressure-unavailable-test"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def hold_summary(run_id: str, gate_reason: str = "test_hold") -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": gate_reason,
        "approved_for_live": False,
        "approved_for_model_training": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }


class Phase51HIPressureUnavailableMatrixTest(unittest.TestCase):
    def test_pressure_unavailable_is_distinct_hold_not_unknown_or_observed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            observed = tmp / "observed_pfill"
            quarantine = tmp / "quarantine"
            canonical = tmp / "canonical"
            queue = tmp / "queue_churn"
            markout = tmp / "markout"
            h_root = tmp / "phase51h"
            i_root = tmp / "phase51i"

            write_json(
                observed / "pfill_outcome_summary.json",
                {
                    **hold_summary("observed_pfill", "observed_source_owner_pfill_test"),
                    "order_label_count": 1,
                    "censored_count": 0,
                    "excluded_quarantine_count": 0,
                    "excluded_quarantine_reason_counts": {},
                    "observed_only_pack_warning": "single-row governance test fixture",
                },
            )
            write_jsonl(
                observed / "pfill_order_labels.jsonl",
                [
                    {
                        "schema_version": 1,
                        "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                        "label_seq": 1,
                        "run_id": "observed_pfill",
                        "baseline_commit": BASELINE_COMMIT,
                        "canonical_group_id": "group-lighter",
                        "order_key": "canonical-lighter",
                        "source_order_keys": ["source-lighter"],
                        "source_label_count": 1,
                        "source_telemetry_sha256": SOURCE_SHA,
                        "venue_id": "lighter",
                        "side": "ASK",
                        "order_holdout_split": "TRAIN",
                        "outcome_status": "OBSERVED_FILLED",
                        "p_fill_outcome": 1.0,
                        "fill_count": 1,
                        "maker_taker_role_counts": {"MAKER": 1, "TAKER": 0, "UNKNOWN": 0},
                        "observed_horizon_source_ticks": 4,
                        "terminal_action_first": "fill",
                        "terminal_event_count": 1,
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                        "admissible_for_ev_admission": False,
                    }
                ],
            )

            write_json(quarantine / "quarantine_review_summary.json", hold_summary("quarantine"))
            write_jsonl(
                quarantine / "source_reconciliation_manifest.jsonl",
                [
                    {
                        "canonical_group_id": "group-lighter",
                        "source_telemetry_sha256": SOURCE_SHA,
                        "venue_id": "lighter",
                        "review_status": "BINARY_OBSERVED_FILLED_DIAGNOSTIC",
                        "included_in_observed_only_pack": True,
                    }
                ],
            )
            write_jsonl(
                canonical / "source_to_canonical_order_manifest.jsonl",
                [
                    {
                        "canonical_group_id": "group-lighter",
                        "canonical_order_key": "canonical-lighter",
                        "source_order_key": "source-lighter",
                        "source_telemetry_sha256": SOURCE_SHA,
                    }
                ],
            )

            write_json(
                queue / "queue_churn_summary.json",
                {
                    **hold_summary("queue", "pressure_unavailable_governance_hold"),
                    "source_telemetry_sha256": SOURCE_SHA,
                },
            )
            write_jsonl(
                queue / "queue_churn_labels.jsonl",
                [
                    {
                        "schema_version": 1,
                        "label_type": "QUEUE_CHURN_LABEL",
                        "label_seq": 1,
                        "run_id": "queue",
                        "baseline_commit": BASELINE_COMMIT,
                        "source_telemetry_sha256": SOURCE_SHA,
                        "order_key": "source-lighter",
                        "venue_id": "lighter",
                        "side": "ASK",
                        "lifecycle_join_status": "JOINED",
                        "queue_reset_proxy_event_count": 0,
                        "replace_event_count": 0,
                        "cancel_event_count": 1,
                        "cancel_all_event_count": 0,
                        "churn_event_count": 1,
                        "native_limit_pressure_status": "PRESSURE_UNAVAILABLE_GOVERNANCE_HOLD",
                        "approved_for_live": False,
                        "approved_for_model_training": False,
                        "admissible_for_ev_admission": False,
                    }
                ],
            )
            write_json(
                markout / "markout_calibration_readiness_summary.json",
                {
                    **hold_summary("markout", "single_row_markout_readiness_context"),
                    "source_telemetry_sha256_list": [SOURCE_SHA],
                },
            )

            h_run = phase51h.build_feature_audit(
                observed_pfill_run=observed,
                quarantine_review_run=quarantine,
                canonical_pfill_run=canonical,
                queue_churn_runs=[queue],
                markout_readiness_runs=[markout],
                horizon_recovery_run=None,
                filled_horizon_recovery_run=None,
                filled_horizon_source_key_recovery_run=None,
                maker_taker_recovery_run=None,
                output_root=h_root,
                run_id="phase51h_pressure_unavailable",
                timestamp_ns=1700000000000000000,
                min_observed_per_bucket=1,
                min_holdout_observed_per_bucket=0,
            )
            h_summary = json.loads((h_run / "pfill_feature_audit_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(h_summary["gate_reason"], "phase51h_lighter_native_limit_pressure_unavailable_governance_hold")
            self.assertEqual(h_summary["native_limit_pressure_unavailable_count"], 1)
            self.assertEqual(h_summary["native_limit_unknown_count"], 0)
            self.assertEqual(h_summary["native_limit_observed_count"], 0)
            self.assertFalse(h_summary["approved_for_model_training"])
            self.assertFalse(h_summary["approved_for_live"])

            h_label = json.loads((h_run / "pfill_feature_coverage_labels.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(h_label["native_limit_pressure_status"], "UNAVAILABLE_GOVERNANCE_HOLD")
            self.assertIn("lighter_native_limit_pressure_unavailable_governance_hold", h_label["missing_features"])
            self.assertNotIn("lighter_native_limit_pressure_not_observed", h_label["missing_features"])
            self.assertTrue(h_label["observed_horizon_available"])

            i_run = phase51i.build_matrix_admissibility(
                feature_audit_run=h_run,
                output_root=i_root,
                run_id="phase51i_pressure_unavailable",
                timestamp_ns=1700000000000001000,
            )
            i_summary = json.loads((i_run / "pfill_feature_matrix_admissibility_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(i_summary["gate_reason"], "phase51i_lighter_native_limit_pressure_unavailable_governance_hold")
            self.assertIn(
                "lighter_native_limit_pressure_unavailable_governance_hold",
                i_summary["matrix_blocker_ids"],
            )
            self.assertNotIn("lighter_native_limit_pressure_not_fully_observed", i_summary["matrix_blocker_ids"])
            self.assertEqual(i_summary["native_limit_pressure_unavailable_count"], 1)
            self.assertEqual(i_summary["native_limit_unknown_count"], 0)
            self.assertEqual(i_summary["native_limit_observed_count"], 0)
            self.assertFalse(i_summary["approved_for_model_training"])
            self.assertFalse(i_summary["approved_for_live"])

            blockers = [
                json.loads(line)
                for line in (i_run / "pfill_feature_matrix_blockers.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            unavailable = next(
                blocker
                for blocker in blockers
                if blocker["blocker_id"] == "lighter_native_limit_pressure_unavailable_governance_hold"
            )
            self.assertEqual(unavailable["measured_count"], 1)
            self.assertEqual(unavailable["gate_status"], "HOLD")
            self.assertFalse(unavailable["approved_for_model_training"])
            self.assertFalse(unavailable["approved_for_live"])


if __name__ == "__main__":
    unittest.main()
