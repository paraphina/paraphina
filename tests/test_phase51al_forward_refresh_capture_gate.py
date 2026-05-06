import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


class TestPhase51alForwardRefreshCaptureGate(unittest.TestCase):
    def test_materializes_forward_refresh_pack_and_phase51ak_recovers_targets(self):
        script_dir = Path(__file__).resolve().parents[1]
        phase51al_tool = script_dir / "tools" / "phase51al_forward_refresh_capture_gate.py"
        phase51ak_tool = script_dir / "tools" / "phase51ak_blocker_resolution_runner.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_jsonl = tmp_path / "forward_refresh.jsonl"
            write_jsonl(
                input_jsonl,
                [
                    {
                        "target_type": "native_role",
                        "venue_id": "extended",
                        "canonical_group_id": "extended-forward-refresh-role",
                        "order_key": "extended-forward-refresh-order",
                        "client_order_id_hash": "hash-client-extended",
                        "order_id_hash": "hash-order-extended",
                        "isTaker": False,
                        "no_live_flag": True,
                        "approved_for_live": False,
                        "live_orders_allowed": False,
                    },
                    {
                        "target_type": "lighter_native_limit",
                        "venue_id": "lighter",
                        "canonical_group_id": "lighter-forward-refresh-pressure",
                        "order_key": "lighter-forward-refresh-order",
                        "client_order_id_hash": "hash-client-lighter",
                        "active_order_headroom_account": 95,
                        "active_order_headroom_market": 12,
                        "sendtx_per_minute_limit": 60,
                        "sendtx_per_minute_remaining": 58,
                        "rest_requests_per_minute_limit": 120,
                        "rest_requests_per_minute_remaining": 118,
                        "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                        "no_live_flag": True,
                        "approved_for_live": False,
                        "live_orders_allowed": False,
                    },
                ],
            )

            output_root = tmp_path / "phase51al_runs"
            al_result = subprocess.run(
                [
                    sys.executable,
                    str(phase51al_tool),
                    "--input-jsonl",
                    str(input_jsonl),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51al_forward_refresh_fixture",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(al_result.returncode, 0, f"stdout={al_result.stdout}\nstderr={al_result.stderr}")

            al_run = output_root / "phase51al_forward_refresh_fixture"
            summary = json.loads((al_run / "phase51al_forward_refresh_capture_summary.json").read_text())
            self.assertEqual(summary["baseline_commit"], BASELINE_COMMIT)
            self.assertEqual(summary["gate_status"], "HOLD")
            self.assertEqual(summary["native_role_capture_target_count"], 1)
            self.assertEqual(summary["lighter_native_limit_capture_target_count"], 1)
            self.assertFalse(summary["live_orders_allowed"])

            ak_output_root = tmp_path / "phase51ak_runs"
            ak_result = subprocess.run(
                [
                    sys.executable,
                    str(phase51ak_tool),
                    "--target-run",
                    str(al_run / "target_run"),
                    "--request-pack",
                    str(al_run / "phase51al_request_pack"),
                    "--no-default-current-manifest",
                    "--candidate-manifest",
                    str(al_run / "candidate_manifest.forward_refresh.json"),
                    "--target-pack-mode",
                    "forward-refresh",
                    "--output-root",
                    str(ak_output_root),
                    "--run-id",
                    "phase51ak_forward_refresh_fixture",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(ak_result.returncode, 0, f"stdout={ak_result.stdout}\nstderr={ak_result.stderr}")

            ak_run = ak_output_root / "phase51ak_forward_refresh_fixture"
            ak_summary = json.loads((ak_run / "phase51ak_blocker_resolution_summary.json").read_text())
            self.assertTrue(ak_summary["phase51v_downstream_chain_ready"])
            self.assertFalse(ak_summary["forward_refresh_required"])
            self.assertEqual(ak_summary["target_pack_mode"], "forward-refresh")
            self.assertEqual(ak_summary["decision_status_counts"], {"READY_FORWARD_REFRESH_PACK": 2})

    def test_rejects_raw_identifier_fields(self):
        script_dir = Path(__file__).resolve().parents[1]
        phase51al_tool = script_dir / "tools" / "phase51al_forward_refresh_capture_gate.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_jsonl = tmp_path / "raw_identifier.jsonl"
            write_jsonl(
                input_jsonl,
                [
                    {
                        "target_type": "native_role",
                        "venue_id": "extended",
                        "canonical_group_id": "bad-target",
                        "order_key": "bad-order",
                        "order_id": "raw-order-id",
                        "isTaker": False,
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(phase51al_tool),
                    "--input-jsonl",
                    str(input_jsonl),
                    "--output-root",
                    str(tmp_path / "runs"),
                    "--run-id",
                    "phase51al_reject_raw",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("raw identifier", result.stderr)

    def test_rejects_incomplete_lighter_pressure_rows(self):
        script_dir = Path(__file__).resolve().parents[1]
        phase51al_tool = script_dir / "tools" / "phase51al_forward_refresh_capture_gate.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_jsonl = tmp_path / "incomplete_pressure.jsonl"
            write_jsonl(
                input_jsonl,
                [
                    {
                        "target_type": "lighter_native_limit",
                        "venue_id": "lighter",
                        "canonical_group_id": "lighter-missing-pressure",
                        "order_key": "lighter-missing-pressure-order",
                        "active_order_headroom_account": 95,
                        "active_order_headroom_market": 12,
                        "sendtx_per_minute_limit": 60,
                        "sendtx_per_minute_remaining": 58,
                        "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(phase51al_tool),
                    "--input-jsonl",
                    str(input_jsonl),
                    "--output-root",
                    str(tmp_path / "runs"),
                    "--run-id",
                    "phase51al_reject_incomplete_pressure",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("REST-or-weighted", result.stderr)


if __name__ == "__main__":
    unittest.main()
