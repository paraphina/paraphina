import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


class TestPhase51anSourceOwnerForwardRefreshCapture(unittest.TestCase):
    def test_empty_inbox_emits_waiting_contract_and_safe_intake(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51an_source_owner_forward_refresh_capture.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            inbox = tmp_path / "source_owner_inbox" / "phase51"

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--inbox",
                    str(inbox),
                    "--output-root",
                    str(tmp_path / "phase51an_runs"),
                    "--phase51al-output-root",
                    str(tmp_path / "phase51al_runs"),
                    "--run-id",
                    "phase51an_empty",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--update-intake-manifest",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = tmp_path / "phase51an_runs" / "phase51an_empty"
            summary = json.loads((run_dir / "phase51an_source_owner_forward_refresh_capture_summary.json").read_text())
            self.assertEqual(summary["baseline_commit"], BASELINE_COMMIT)
            self.assertEqual(summary["control_status"], "AWAITING_SOURCE_OWNER_ROWS")
            self.assertEqual(summary["forward_refresh_row_count"], 0)
            self.assertFalse(summary["live_orders_allowed"])
            self.assertTrue((inbox / "forward_refresh.jsonl").exists())
            self.assertTrue((inbox / "SOURCE_OWNER_CAPTURE_CONTRACT.md").exists())

            intake = json.loads((inbox / "intake.json").read_text())
            self.assertEqual(intake["phase51al_summaries"], [])
            self.assertEqual(intake["phase51aj_source_json"], [])
            self.assertEqual(intake["phase51ab_pressure_jsonls"], [])
            self.assertTrue(intake["no_live_flag"])
            self.assertFalse(intake["approved_for_live"])

    def test_nonempty_forward_refresh_materializes_phase51al_and_intake_manifest(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51an_source_owner_forward_refresh_capture.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            inbox = tmp_path / "source_owner_inbox" / "phase51"
            write_jsonl(
                inbox / "forward_refresh.jsonl",
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
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--inbox",
                    str(inbox),
                    "--output-root",
                    str(tmp_path / "phase51an_runs"),
                    "--phase51al-output-root",
                    str(tmp_path / "phase51al_runs"),
                    "--run-id",
                    "phase51an_ready",
                    "--phase51al-run-id",
                    "phase51al_ready",
                    "--timestamp-ns",
                    "1700000000000000000",
                    "--update-intake-manifest",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = tmp_path / "phase51an_runs" / "phase51an_ready"
            summary = json.loads((run_dir / "phase51an_source_owner_forward_refresh_capture_summary.json").read_text())
            self.assertEqual(summary["control_status"], "PHASE51AL_FORWARD_REFRESH_PACK_MATERIALIZED")
            self.assertEqual(summary["forward_refresh_row_count"], 1)
            self.assertEqual(summary["forward_refresh_row_counts_by_target_type"], {"native_role": 1})
            self.assertTrue(summary["phase51al_summary_path"])

            al_summary_path = Path(summary["phase51al_summary_path"])
            al_summary = json.loads(al_summary_path.read_text())
            self.assertEqual(al_summary["native_role_capture_target_count"], 1)
            self.assertEqual(al_summary["lighter_native_limit_capture_target_count"], 0)
            self.assertFalse(al_summary["live_orders_allowed"])

            intake = json.loads((inbox / "intake.json").read_text())
            self.assertEqual(intake["phase51al_summaries"], [str(al_summary_path)])
            self.assertEqual(intake["phase51aj_source_json"], [])
            self.assertEqual(intake["phase51ab_pressure_jsonls"], [])

    def test_rejects_raw_identifier_fields_before_materializing_phase51al(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51an_source_owner_forward_refresh_capture.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            inbox = tmp_path / "source_owner_inbox" / "phase51"
            write_jsonl(
                inbox / "forward_refresh.jsonl",
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
                    str(tool_path),
                    "--inbox",
                    str(inbox),
                    "--output-root",
                    str(tmp_path / "phase51an_runs"),
                    "--phase51al-output-root",
                    str(tmp_path / "phase51al_runs"),
                    "--run-id",
                    "phase51an_reject_raw",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("raw identifier", result.stderr)
            self.assertFalse((tmp_path / "phase51al_runs" / "phase51al_reject_raw").exists())


if __name__ == "__main__":
    unittest.main()
