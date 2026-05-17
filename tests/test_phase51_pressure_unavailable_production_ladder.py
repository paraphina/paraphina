import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def unavailable_packet(**overrides):
    packet = {
        "schema_version": 1,
        "producer": "Phase51LighterPressureSource",
        "target_type": "lighter_native_limit",
        "venue_id": "lighter",
        "run_id": "governance-closeout",
        "gate_status": "HOLD",
        "native_limit_event_time_status": "PRESSURE_UNAVAILABLE",
        "active_order_provenance_state": "AUDITED_EXPLICIT_SOURCE_UNAVAILABLE",
        "sendtx_provenance_state": "AUDITED_EXPLICIT_SOURCE_UNAVAILABLE",
        "request_pressure_provenance_state": "AUDITED_EXPLICIT_SOURCE_UNAVAILABLE",
        "pressure_packet_state": "AUDITED_SANITIZED_PRESSURE_UNAVAILABLE",
        "pressure_state": "pressure_unavailable",
        "raw_identifier_redaction_status": "PASS",
        "fixture_provenance": "SANITIZED_GOVERNANCE_CLOSEOUT",
        "native_limit_pressure_source": "LIGHTER_SOURCE_ROUTE_CLOSED_NEGATIVE",
        "transform_version": "governance-transform-v1",
        "redaction_policy_version": "governance-redaction-v1",
        "source_count": 5,
        "account_limits_probe_status": "REQUIRED_DIMENSIONS_ABSENT",
        "passive_sendtx_observation_status": "REQUIRED_DIMENSIONS_ABSENT",
        "repo_docs_sdk_audit_status": "NO_EXPLICIT_SOURCE_FOUND",
        "websocket_schema_audit_status": "NO_COMPLETE_PRESSURE_DIMENSIONS",
        "pressure_unavailable_reason": "LIGHTER_EXPLICIT_PRESSURE_SOURCE_CLOSED_NEGATIVE",
        "governance_decision_sha256": "a" * 64,
        "completeness_flag": False,
        "is_synthetic_fixture": False,
        "derived_from_real_evidence": True,
        "runtime_observation": False,
        "capture_enabled": False,
        "gap_or_staleness_flag": True,
        "missing_pressure_values_inferred": False,
        "volume_quota_substitute_rejected": True,
        "no_live_flag": True,
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "approved_for_financial_claim": False,
        "admissible_for_model_training": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "blocker_cleared": False,
    }
    packet.update(overrides)
    return packet


class Phase51PressureUnavailableProductionLadderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Path(__file__).resolve().parents[1]

    def _target_run(self, root: Path) -> Path:
        target_run = root / "target_run"
        target_run.mkdir()
        write_json(
            target_run / "phase51u_forward_capture_target_manifest_summary.json",
            {
                "schema_version": 1,
                "baseline_commit": BASELINE_COMMIT,
                "gate_status": "HOLD",
                "no_live_flag": True,
                "approved_for_live": False,
                "live_orders_allowed": False,
            },
        )
        write_jsonl(target_run / "native_role_capture_targets.jsonl", [])
        write_jsonl(
            target_run / "lighter_native_limit_capture_targets.jsonl",
            [
                {
                    "schema_version": 1,
                    "baseline_commit": BASELINE_COMMIT,
                    "venue_id": "lighter",
                    "canonical_group_id": "lighter-limit-target",
                    "order_key": "lighter-limit-order-key",
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                }
            ],
        )
        return target_run

    def _observed_pfill_run(self, root: Path) -> Path:
        observed_run = root / "observed_pfill"
        observed_run.mkdir()
        write_json(
            observed_run / "pfill_outcome_summary.json",
            {
                "schema_version": 1,
                "baseline_commit": BASELINE_COMMIT,
                "gate_status": "HOLD",
                "order_label_count": 1,
                "no_live_flag": True,
                "approved_for_live": False,
                "live_orders_allowed": False,
            },
        )
        write_jsonl(
            observed_run / "pfill_order_labels.jsonl",
            [
                {
                    "schema_version": 1,
                    "label_type": "ORDER_PFILL_OUTCOME_LABEL",
                    "baseline_commit": BASELINE_COMMIT,
                    "canonical_group_id": "lighter-limit-target",
                    "order_key": "lighter-limit-order-key",
                    "venue_id": "lighter",
                    "fill_count": 1,
                    "maker_taker_role_counts": {"MAKER": 1, "TAKER": 0, "UNKNOWN": 0},
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                }
            ],
        )
        return observed_run

    def _candidate_manifest(self, root: Path, source_path: Path) -> Path:
        manifest = root / "candidate_manifest.json"
        write_json(
            manifest,
            {
                "manifest_version": 1,
                "baseline_commit": BASELINE_COMMIT,
                "no_live_flag": True,
                "approved_for_live": False,
                "live_orders_allowed": False,
                "sources": [
                    {
                        "source_id": "lighter_pressure_unavailable",
                        "venue_id": "lighter",
                        "path": str(source_path),
                    }
                ],
                "source_links": [],
            },
        )
        return manifest

    def test_phase51v_reports_pressure_unavailable_without_ready_or_false_clearance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target_run = self._target_run(tmp)
            unavailable_source = tmp / "lighter_pressure_unavailable.jsonl"
            write_jsonl(unavailable_source, [unavailable_packet()])
            manifest = self._candidate_manifest(tmp, unavailable_source)
            output_root = tmp / "phase51v"

            result = subprocess.run(
                [
                    sys.executable,
                    str(self.repo / "tools" / "phase51v_forward_capture_bundle_readiness.py"),
                    "--target-run",
                    str(target_run),
                    "--candidate-manifest",
                    str(manifest),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51v_pressure_unavailable",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
            run_dir = output_root / "phase51v_pressure_unavailable"
            summary = json.loads((run_dir / "phase51v_forward_capture_bundle_readiness_summary.json").read_text())
            self.assertEqual(summary["lighter_native_limit_capture_target_ready_count"], 0)
            self.assertEqual(summary["lighter_native_limit_capture_target_missing_count"], 1)
            self.assertEqual(summary["lighter_native_limit_pressure_unavailable_source_count"], 1)
            self.assertEqual(summary["lighter_native_limit_pressure_unavailable_target_count"], 1)
            self.assertFalse(summary["downstream_chain_ready"])
            self.assertFalse(summary["clears_phase51_blockers"])
            self.assertFalse(summary["revised_pressure_unavailable_contract_clears_blocker"])

            labels_text = (run_dir / "capture_bundle_readiness_labels.jsonl").read_text(encoding="utf-8")
            self.assertIn('"lighter_limit_pressure_state":"pressure_unavailable"', labels_text)
            self.assertIn('"lighter_limit_governance_status":"PRESSURE_UNAVAILABLE_GOVERNANCE_HOLD"', labels_text)
            self.assertNotIn('"lighter_limit_target_ready":true', labels_text)

    def test_phase51q_reports_pressure_unavailable_as_hold_not_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            observed_run = self._observed_pfill_run(tmp)
            unavailable_source = tmp / "lighter_pressure_unavailable.jsonl"
            write_jsonl(unavailable_source, [unavailable_packet()])
            output_root = tmp / "phase51q"

            result = subprocess.run(
                [
                    sys.executable,
                    str(self.repo / "tools" / "phase51q_forward_native_evidence_capture.py"),
                    "--observed-pfill-run",
                    str(observed_run),
                    "--native-limit-jsonl",
                    str(unavailable_source),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51q_pressure_unavailable",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
            run_dir = output_root / "phase51q_pressure_unavailable"
            summary = json.loads((run_dir / "phase51q_forward_native_evidence_summary.json").read_text())
            self.assertEqual(
                summary["native_limit_pressure_status_counts"],
                {"PRESSURE_UNAVAILABLE_GOVERNANCE_HOLD": 1},
            )
            self.assertEqual(summary["native_limit_pressure_unavailable_count"], 1)
            self.assertEqual(summary["gate_reason"], "phase51q_forward_native_evidence_incomplete")
            self.assertFalse(summary["revised_pressure_unavailable_contract_clears_blocker"])

            labels_text = (run_dir / "native_limit_pressure_labels.jsonl").read_text(encoding="utf-8")
            self.assertIn('"pressure_state":"pressure_unavailable"', labels_text)
            self.assertNotIn("volume_quota_remaining", labels_text)

    def test_phase51ak_marks_lighter_target_as_pressure_unavailable_governance_hold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target_run = self._target_run(tmp)
            request_pack = tmp / "request_pack"
            request_pack.mkdir()
            write_json(request_pack / "manifest.json", {"schema_version": 1, "baseline_commit": BASELINE_COMMIT})
            write_jsonl(request_pack / "source_link_request_targets.jsonl", [])
            write_jsonl(request_pack / "source_link_request_sources.jsonl", [])
            unavailable_source = tmp / "lighter_pressure_unavailable.jsonl"
            write_jsonl(unavailable_source, [unavailable_packet()])
            manifest = self._candidate_manifest(tmp, unavailable_source)
            output_root = tmp / "phase51ak"

            result = subprocess.run(
                [
                    sys.executable,
                    str(self.repo / "tools" / "phase51ak_blocker_resolution_runner.py"),
                    "--target-run",
                    str(target_run),
                    "--request-pack",
                    str(request_pack),
                    "--no-default-current-manifest",
                    "--candidate-manifest",
                    str(manifest),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ak_pressure_unavailable",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")
            run_dir = output_root / "phase51ak_pressure_unavailable"
            summary = json.loads((run_dir / "phase51ak_blocker_resolution_summary.json").read_text())
            self.assertEqual(summary["gate_reason"], "phase51ak_pressure_unavailable_governance_hold_nonlive")
            self.assertEqual(summary["lighter_native_limit_pressure_unavailable_target_count"], 1)
            self.assertFalse(summary["phase51v_downstream_chain_ready"])
            self.assertFalse(summary["forward_refresh_required"])
            self.assertFalse(summary["clears_phase51_blockers"])
            self.assertFalse(summary["revised_pressure_unavailable_contract_clears_blocker"])

            decisions = [
                json.loads(line)
                for line in (run_dir / "phase51ak_blocker_target_decisions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["decision_status"], "PRESSURE_UNAVAILABLE_GOVERNANCE_HOLD")
            self.assertEqual(
                decisions[0]["next_required_action"],
                "APPLY_REVISED_PRESSURE_UNAVAILABLE_GOVERNANCE_CONTRACT",
            )
            self.assertFalse(decisions[0]["current_pack_target_ready"])
            self.assertFalse(decisions[0]["forward_refresh_required"])
            self.assertTrue(decisions[0]["pressure_unavailable_governance_hold"])


if __name__ == "__main__":
    unittest.main()
