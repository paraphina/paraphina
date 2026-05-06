import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


class TestPhase51akBlockerResolutionRunner(unittest.TestCase):
    def _write_target_run(self, target_run: Path, role_targets: list[dict], limit_targets: list[dict]) -> None:
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
        write_jsonl(target_run / "native_role_capture_targets.jsonl", role_targets)
        write_jsonl(target_run / "lighter_native_limit_capture_targets.jsonl", limit_targets)

    def _write_request_pack(self, request_pack: Path, targets: list[dict]) -> None:
        request_pack.mkdir()
        write_json(request_pack / "manifest.json", {"schema_version": 1, "baseline_commit": BASELINE_COMMIT})
        write_jsonl(request_pack / "source_link_request_targets.jsonl", targets)
        write_jsonl(request_pack / "source_link_request_sources.jsonl", [])

    def _role_target(self, client_id: str) -> dict:
        return {
            "schema_version": 1,
            "baseline_commit": BASELINE_COMMIT,
            "venue_id": "extended",
            "canonical_group_id": "extended-target",
            "order_key": "extended-order-key",
            "client_order_id_hash": stable_hash(client_id),
            "order_id_hash": None,
            "no_live_flag": True,
            "approved_for_live": False,
            "live_orders_allowed": False,
        }

    def _limit_target(self) -> dict:
        return {
            "schema_version": 1,
            "baseline_commit": BASELINE_COMMIT,
            "venue_id": "lighter",
            "canonical_group_id": "lighter-limit-target",
            "order_key": "lighter-limit-order-key",
            "no_live_flag": True,
            "approved_for_live": False,
            "live_orders_allowed": False,
        }

    def test_recovers_current_pack_with_direct_role_and_lighter_pressure_sources(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51ak_blocker_resolution_runner.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "target_run"
            request_pack = tmp_path / "request_pack"
            output_root = tmp_path / "runs"
            external_id = "extended-client-direct-001"
            role_target = self._role_target(external_id)
            limit_target = self._limit_target()
            self._write_target_run(target_run, [role_target], [limit_target])
            self._write_request_pack(request_pack, [role_target])

            extended_source = tmp_path / "extended.json"
            write_json(
                extended_source,
                {
                    "data": [
                        {
                            "type": "TRADE",
                            "externalOrderId": external_id,
                            "orderId": "9223372036854775808",
                            "isTaker": False,
                        }
                    ]
                },
            )
            pressure_source = tmp_path / "lighter_pressure.jsonl"
            write_jsonl(
                pressure_source,
                [
                    {
                        "schema_version": "1",
                        "baseline_commit": BASELINE_COMMIT,
                        "venue_id": "lighter",
                        "canonical_group_id": "lighter-limit-target",
                        "order_key": "lighter-limit-order-key",
                        "active_order_headroom_account": 90,
                        "active_order_headroom_market": 9,
                        "sendtx_per_minute_limit": 60,
                        "sendtx_per_minute_remaining": 59,
                        "rest_requests_per_minute_limit": 120,
                        "rest_requests_per_minute_remaining": 119,
                        "native_limit_event_time_status": "EVENT_TIME_ALIGNED",
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
                    "--target-run",
                    str(target_run),
                    "--request-pack",
                    str(request_pack),
                    "--no-default-current-manifest",
                    "--phase51aj-source-json",
                    f"extended={extended_source}",
                    "--phase51ab-pressure-jsonl",
                    str(pressure_source),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ak_ready",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = output_root / "phase51ak_ready"
            summary = json.loads((run_dir / "phase51ak_blocker_resolution_summary.json").read_text())
            self.assertTrue(summary["phase51v_downstream_chain_ready"])
            self.assertFalse(summary["forward_refresh_required"])
            self.assertEqual(summary["decision_status_counts"], {"RECOVERED_CURRENT_PACK": 2})

            decisions = [
                json.loads(line)
                for line in (run_dir / "phase51ak_blocker_target_decisions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual({row["decision_status"] for row in decisions}, {"RECOVERED_CURRENT_PACK"})
            self.assertEqual({row["target_type"] for row in decisions}, {"native_role", "lighter_native_limit"})

    def test_marks_missing_current_pack_targets_as_forward_refresh_required(self):
        script_dir = Path(__file__).resolve().parents[1]
        tool_path = script_dir / "tools" / "phase51ak_blocker_resolution_runner.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            target_run = tmp_path / "target_run"
            request_pack = tmp_path / "request_pack"
            output_root = tmp_path / "runs"
            role_target = self._role_target("missing-client")
            self._write_target_run(target_run, [role_target], [])
            self._write_request_pack(request_pack, [role_target])

            empty_source = tmp_path / "empty_source.jsonl"
            write_jsonl(empty_source, [])
            candidate_manifest = tmp_path / "candidate_manifest.json"
            write_json(
                candidate_manifest,
                {
                    "manifest_version": 1,
                    "baseline_commit": BASELINE_COMMIT,
                    "no_live_flag": True,
                    "approved_for_live": False,
                    "live_orders_allowed": False,
                    "sources": [
                        {
                            "source_id": "empty_extended_source",
                            "venue_id": "extended",
                            "path": str(empty_source),
                        }
                    ],
                    "source_links": [],
                },
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(tool_path),
                    "--target-run",
                    str(target_run),
                    "--request-pack",
                    str(request_pack),
                    "--no-default-current-manifest",
                    "--candidate-manifest",
                    str(candidate_manifest),
                    "--output-root",
                    str(output_root),
                    "--run-id",
                    "phase51ak_hold",
                    "--timestamp-ns",
                    "1700000000000000000",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"stdout={result.stdout}\nstderr={result.stderr}")

            run_dir = output_root / "phase51ak_hold"
            summary = json.loads((run_dir / "phase51ak_blocker_resolution_summary.json").read_text())
            self.assertFalse(summary["phase51v_downstream_chain_ready"])
            self.assertTrue(summary["forward_refresh_required"])
            self.assertEqual(summary["next_required_action"], "obtain_validated_mapping_or_forward_refresh_target_pack_with_event_time_sources")

            decisions = [
                json.loads(line)
                for line in (run_dir / "phase51ak_blocker_target_decisions.jsonl").read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(decisions), 1)
            self.assertEqual(decisions[0]["decision_status"], "UNRECOVERABLE_FROM_LOCAL_ARTIFACTS")
            self.assertEqual(decisions[0]["next_required_action"], "FORWARD_REFRESH_REQUIRED")


if __name__ == "__main__":
    unittest.main()
